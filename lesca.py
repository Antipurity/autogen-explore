"""
Learnable Scaffolding, for making learnable choices in arbitrary programs.

(Unless in coroutines, or direct execution stack manipulation.)

Contains:

- Learnable scaffolding:
  - scaffolding;
  - static, state;
  - num, choice, static_choice;
  - impure, cached, replay.

- Gradient-related stuff (since we re-implement machine learning with NumPy):
  - var, leaky_relu, dense, loss2;
  - Struct, AutoFunc, args;
  - _default_nn, _renn;
  - ExecState, SetExecState, GetExecState, adjust.

- Utility:
  - parallel, persist;
  - attr;
  - Limit, _LimitExceeded.

A quick example:

```
import lesca as ls

static = ls.scaffolding(feature_size = 3)
ctx = static(x = ls.choice())
@ls.cached
@ls.replay()
@static()
def f3(a):
    c = ls.state.x(0,1)
    return a+1 if c == 0 else a-1

with ctx:
    ls.state.x = f3(5)
    # The choice in `f3` will (try to) maximize this value.
    #   Repeat this many times to learn, and/or
    #     use `ls.replay.redo(f3)` to learn.
```
"""
import numpy as np
import threading # local
import time
import weakref
import random



TL = threading.local() # Thread-local storage for execution-state-related global variables.

TL._outer_node = None
TL._sub = None # dict from id(TL._outer_node) to
#   None or [id(func), TL._sub, exec_state, <repeating>]
TL.limits = []
TL._Seen_current = None
TL._Replay = None
TL._Replay_state = None
TL.var_deltas = {}
TL.cur_scaffold = None



class Struct(list):
    """Immutable arrays, mostly used for specifying function body DAGs."""
    def __init__(self, *head_then_args):
        super().__init__(head_then_args)



def _get_outer_node(): return TL._outer_node
def _set_outer_node(to): TL._outer_node = to

def _fallthrough(x):
    return x



class SetExecState:
    """A context manager for remembering execution state that needs to be passed to the adjuster.
    If this is used, then GetExecState must also be used in the adjuster.
    See ExecState."""
    __slots__ = 'func', 'state', 'outer_sub', 'i'
    def __init__(self, func, state):
        self.func = func
        self.state = state
    def __enter__(self):
        self.outer_sub, TL._sub = TL._sub, {}
        self.i = id(TL._outer_node)
        return self.state
    def __exit__(self,x,y,z):
        os, i = self.outer_sub, self.i
        if os is None: return
        if i not in os:
            os[i] = []
        os[i].append(id(self.func))
        os[i].append(TL._sub)
        os[i].append(self.state)
        TL._sub = os

class GetExecState:
    """A context manager for remembering execution state in the adjuster after SetExecState.

    See ExecState."""
    __slots__ = 'func', 'outer_sub'
    def __init__(self, func):
        self.func = func
    def __enter__(self):
        i, os = id(TL._outer_node), TL._sub
        self.outer_sub = os
        if i in os:
            j = len(os[i]) - 3
            if os[i][j] == id(self.func):
                # Hoping that subfunctions are unique is the best we can hope for
                #   without analyzing Python's internal representation of functions.
                #     (Or requiring perfect reversal of execution order.)
                TL._sub = os[i][j+1]
                state = os[i][j+2]
                del os[i][j:j+3]
                return state
        print('When looking for', self.func, 'id', id(self.func))
        print('  in outer node:', TL._outer_node, 'id', i)
        print('  in:')
        for k in os:
            print(str(k)+':', os[k])
        raise AssertionError('GetExecState that did not have SetExecState, or had a wrong one afterwards')
    def __exit__(self,x,y,z):
        TL._sub = self.outer_sub

class ExecState:
    """A context manager that allows connecting execution and adjustment.

        with ExecState():
            with SetExecState(x, {}) as state:
                # execute x
            with GetExecState(x) as state:
                # adjust x
    """
    __slots__ = 's', 'n'
    def __enter__(self):
        self.s, TL._sub = TL._sub, {}
        self.n, TL._outer_node = TL._outer_node, None
    def __exit__(self,x,y,z):
        TL._sub = self.s
        TL._outer_node = self.n

    @staticmethod
    def will_adjust():
        """Returns whether some caller has sworn on everything they ever held dear
        to adjust after execution."""
        return TL._sub is not None



def attr(**kwargs):
    """A decorator that puts attributes onto a function."""
    def decorate(func):
        for k,v in kwargs.items():
            setattr(func, k, v)
        return func
    return decorate



def _postorder(node):
    """Returns the post-order list of `node`.
    All dependencies of a node at an index occur before it in this list."""
    if not isinstance(node, Struct) and not isinstance(node, dict):
        return

    result = []
    def walk(x, visited):
        if not isinstance(x, Struct): return
        if id(x) in visited: return
        else: visited.add(id(x))
        if _fallthrough(x) is x:
            ## Could also randomize the order of visited children.
            for i in range(1, len(x)):
                walk(x[i], visited)
        else:
            walk(_fallthrough(x), visited)
        result.append(x)

    if isinstance(node, Struct):
        walk(node, set())
    else: # dict, {'a':impl, 'b':impl}
        s = set()
        for k in node:
            walk(node[k], s)
    return result



def adjust(func, ins, out, dout):
    """Composable value-aware function adjustment, via reversing execution.

    After execution, propagates loss of output through insides to inputs.
    Make sure that state variables are declared in the order of their usage, the result being the last."""
    if hasattr(func, 'adjust'):
        return func.adjust(ins, out, dout)



class args:
    """Represents args of the currently-called function."""
    def __new__(cls): return cls._singleton
args._singleton = Struct(args)



def _merge(a,b):
    """When an execution graph node is used N times in an adjusted computation, this is called N-1 times to merge adjustments."""
    if a is None:
        return b
    elif b is None:
        return a
    elif type(a) is list:
        if type(b) is not list and type(b) is not tuple:
            print('Expected list to merge to A, got B:', a, b)
            print('  in node', _get_outer_node())
            assert False, "Not enough list-ness"
        if len(a) != len(b):
            print(a, b, 'in', _get_outer_node())
            raise TypeError('Lengths are unequal')
        for i in range(len(a)):
            a[i] = _merge(a[i], b[i])
        return a
    else:
        return a + b



@attr(adjust = lambda i,o,do: [np.where(i[0] > 0, do, do * 0.001)] if do is not None else None)
def leaky_relu(v):
    """An operation that introduces discontinuity."""
    return np.where(v > 0, v, v * 0.001)



def _dense_adjust(i,o,do):
    # https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/ops/math_grad.py#L1668
    x,w = i
    if x is None:
        x = np.zeros(w.shape[0]); x[0] = 1.
    if do is None:
        return
    dx = np.matmul(do, np.transpose(w))
    dw = np.matmul(np.reshape(x, (w.shape[0], 1)), np.reshape(do, (1, w.shape[1])))
    return dx, dw
@attr(adjust = _dense_adjust)
def dense(x, w):
    """Linearly connects every input to every output.

    The second argument is a random matrix of .shape = (input_size, output_size)."""
    if x is None:
        x = np.zeros(w.shape[0]); x[0] = 1.
    return np.matmul(x, w)
    # If x is always of the shape (n,), then what's below is the same as the line above:
    # return np.tensordot(x, w, axes = (-1,0))



@attr(adjust = lambda i,o,do: (i[0] - i[1], i[1] - i[0]))
def loss2(predicted, got):
    """The most basic loss function."""
    return (predicted - got) * (predicted - got) / 2



def _compile_af(af):
    # I ended up watching too much physics and making this.
    #   Incidentally, I fixed a latent bug in `adjust`.
    #     Wasting time on loving something is great.
    po, ais = af._postorder, af._arg_indexes
    if po is None:
        return
    # Pass in constants as args of the outer function.
    src, consts, const_names = ['def outer(.·¯):'], {}, {}
    def const(x):
        if id(x) in const_names: return const_names[id(x)]
        name = 'c' + str(len(consts))
        assert name not in consts, "Overwriting a name"
        consts[name] = x
        const_names[id(x)] = name
        return name
    def at(i, s = 's'): return s + '[' + str(i) + ']'
    def assign(i, *strs):
        to = (str(i) + ' = ') if i is not None else ''
        src.append('\n    ' + to + ''.join(strs))
    af_name = const(af)

    # Execute SSA statements:
    src.append(f'\n def inner(ins):')
    src.append(f'\n  with {const(SetExecState)}({af_name}, [None]*{str(len(po))}) as s:')
    src.append(f'\n   prev_outer_node = {const(_get_outer_node)}()')
    src.append(f'\n   try:')
    for i in range(len(po)):
        x, ai = po[i], ais[i]
        if isinstance(x, Struct):
            if x[0] is args:
                assign(at(i), 'ins')
            elif _fallthrough(x) is x:
                ins = [
                    at(ai[k]) if ai[k] is not None else const(x[k+1])
                    for k in range(len(ai))]
                assign(None, const(_set_outer_node), '(', const(x), ')')
                assign(at(i), const(x[0].call if hasattr(x[0], 'call') else x[0]), '(', ','.join(ins), ')')
            else:
                assign(at(i), at(ai) if ai is not None else const(_fallthrough(x)))
    src.append(f'\n   finally:')
    src.append(f'\n    {const(_set_outer_node)}(prev_outer_node)')
    src.append(f'\n  return {at(len(po) - 1)}')

    # Adjust SSA statements:
    src.append(f'\n def inner_adjust(ins, out, dout):')
    src.append(f'\n  dins = None')
    src.append(f'\n  ds = [None]*{str(len(po))}')
    src.append(f'\n  ds[{str(len(po)-1)}] = dout')
    src.append(f'\n  with {const(GetExecState)}({af_name}) as s:')
    src.append(f'\n   prev_outer_node = {const(_get_outer_node)}()')
    src.append(f'\n   try:')
    for i in reversed(range(len(po))):
        x, ai = po[i], ais[i]
        if isinstance(x, Struct):
            if x[0] is args:
                assign('dins', at(i, 'ds'))
            elif _fallthrough(x) is x:
                ins = [
                    at(ai[k]) if ai[k] is not None else const(x[k+1])
                    for k in range(len(ai))]
                assign(None, const(_set_outer_node), '(', const(x), ')')
                # Using `[v1, v2, _, v3] = _merge([v1, v2, None, v3], adjust(…))` here.
                #   Could be inlined, but since we don't have access to adjusting funcs and
                #     so can't eliminate tuple creation, that should be JIT's job.
                write_to = '[' + ','.join([at(a, 'ds') if a is not None else '_' for a in ai]) + ']'
                read_from = '[' + ','.join([at(a, 'ds') if a is not None else 'None' for a in ai]) + ']'
                to = f'{const(adjust)}({const(x[0])}, [{",".join(ins)}], {at(i)}, {at(i, "ds")})'
                assign(write_to, f'{const(_merge)}({read_from}, {to})')
            elif ai is not None:
                assign(at(ai, 'ds'), f'{const(_merge)}({at(ai, "ds")}, {at(i, "ds")})')
    src.append(f'\n   finally:')
    src.append(f'\n    {const(_set_outer_node)}(prev_outer_node)')
    src.append(f'\n  return (dins,)')

    src.append(f'\n inner.adjust = inner_adjust')
    src.append(f'\n return inner')
    src[0] = f'def outer({",".join(consts.keys())}):'

    locs = {}
    exec(''.join(src), { '__builtins__':{} }, locs)
    return locs['outer'](*consts.values())



class Limit:
    """A context manager for specifying a limit on some measure.
    `Limit.ok()` can be used inside to assert that no limits were broken."""

    @staticmethod
    def ok():
        """Exists."""
        for L in TL.limits:
            L.check()

    def check(self):
        """Checks that this limit is not broken.
        Returns a number to maximize to make this limit broken less."""
        m = self.measure()
        if m > self.maximum:
            raise _LimitExceeded(f"Limit exceeded: {m}/{self.maximum}", self.measure)
        return -m

    def __init__(self, measure, maximum):
        self.measure = measure
        self.maximum = maximum
    def __enter__(self):
        TL.limits.append(self)
    def __exit__(self,x,y,z):
        assert TL.limits[-1] is self
        TL.limits.pop()
    __slots__ = 'measure', 'maximum'

    # Ideally, these would be visible in random-measure generation too, used intelligently.
    class time:
        """Measures time since creation, in milliseconds."""
        def __init__(self): self.s = time.process_time()
        def __call__(self):   return time.process_time() - self.s
        __slots__ = ['s']

    class struct_bytes_change:
        """Returns the change in count of bytes occupied by `Struct`s."""
        def __init__(self): self.s = _bytes
        def __call__(self):   return _bytes - self.s
        __slots__ = ['s']

    class memory_change:
        # …`sys.getallocatedblocks()` also exists…
        """Returns the change in whole kilobytes."""
        def __init__(self): self.s = self.m()
        def __call__(self):   return self.m() - self.s
        __slots__ = ['s']
        def m(_):
            # Somewhat accurate down to kilobytes.
            import resource
            u = resource.getrusage(resource.RUSAGE_SELF)
            mem = u.ru_ixrss + u.ru_idrss + u.ru_isrss
            if not mem:
                mem = u.ru_maxrss
            if sys.platform != 'darwin':
                mem = mem * 1024
            return mem

class _LimitExceeded(RuntimeError):
    """Type of exception raised by `Limit.ok()`."""



class AutoFunc:
    """A dynamically-changeable and adjustable function with one input and one output for symmetry."""
    __slots__ = '_struct', '_postorder', '_arg_indexes', '_compiled', '__weakref__'
    def __init__(self, body):
        self._struct = None
        self._compiled = None
        self._postorder = None
        self._arg_indexes = None
        self._set_body(body)

    def __getstate__(self):
        """Save body for pickling."""
        return self._struct
    def __setstate__(self, state):
        """Restore signature/body/context on unpickling."""
        body = state
        self._struct = self._compiled = self._postorder = self._arg_indexes = None
        self._set_body(body)

    def __call__(self, ins):
        """Executes function body given input."""
        if self._compiled is not None:
            while True:
                Limit.ok()
                try:
                    return self._compiled(ins)
                except _Regenerate:
                    regenerate(self)
        s = self._struct
        while _fallthrough(s) is not s: s = _fallthrough(s)
        return s

    def adjust(self, ins, out, dout):
        """Adjusts function body, exactly reversing execution."""
        if self._compiled is not None:
            return self._compiled.adjust(ins, out, dout)

    def _set_body(self, struct, po = None, ais = None, compiled = None):
        """Sets the executed structure, pre-filling postorder and arg indexes from references, and compiling."""
        if struct is self._struct:
            return
        if po is None and ais is None:
            po = _postorder(struct)
            if po is not None:
                ais = [None] * len(po)
                indexes = { id(po[i]): i for i in range(len(po)) if isinstance(po[i], Struct) }
                for i in range(len(po)):
                    n = po[i]
                    if isinstance(n, Struct):
                        if _fallthrough(n) is n:
                            ai = ais[i] = [None] * (len(n)-1)
                            for j in range(1, len(n)):
                                arg = id(n[j])
                                ai[j-1] = indexes.get(arg)
                                assert indexes.get(arg, -1) < i, 'Cycles in computation'
                        elif id(_fallthrough(n)) in indexes:
                            ais[i] = indexes[id(_fallthrough(n))]
        self._struct, self._postorder, self._arg_indexes = struct, po, ais
        self._compiled = _compile_af(self) if compiled is None else compiled



class _Seen:
    """Context manager in which the same node won't be visited twice (if checking `_Seen.has(map, node)`).

    Basically overridable-by-callers lookup in a dict."""

    @staticmethod
    def exists():
        """Returns True if already in a _Seen context."""
        return TL._Seen_current is not None

    @staticmethod
    def has(m, k):
        """Returns whether the key `k` exists in the dict associated with `m`."""
        c = TL._Seen_current
        if c is not None:
            return id(m) in c and id(k) in c[id(m)]
        return False
    @staticmethod
    def get(m, k):
        """Returns the current value of key `k` in the dict associated with `m`."""
        c = TL._Seen_current
        assert c is not None
        return c[id(m)][id(k)]
    @staticmethod
    def set(m, k, v):
        """Sets the value at key `k` in the dict associated with `m` to be `v`; returns `v`."""
        c = TL._Seen_current
        assert c is not None
        if id(m) not in c: c[id(m)] = {}
        c[id(m)][id(k)] = v
        return v
    @staticmethod
    def pop(m, k):
        """Deletes the value at key `k` in the dict associated with `m` and the dict if needed."""
        c = TL._Seen_current
        assert c is not None
        result = c[id(m)].pop(id(k))
        if len(c[id(m)]) == 0:
            del c[id(m)]
        return result

    __slots__ = 's', 'prev'
    def __init__(self):
        self.s, self.prev = {}, None
    def __enter__(self):
        assert self.prev is None
        self.prev, TL._Seen_current = TL._Seen_current, self.s
    def __exit__(self, x,y,z):
        assert TL._Seen_current is self.s
        TL._Seen_current, self.prev = self.prev, None



def _open_whitebox(af):
    """Returns all function bodies (acyclic structs) of the arg in some self-consistent order."""
    if isinstance(af, AutoFunc):
        return af._struct,



class _renn:
    """
    Creator of estimators of a numeric representation.
    """
    def __init__(self, size, nn):
        self._size = size
        self._nn = nn
        self._single = {}
        self._left = nn(size*2, size*2)
        self._right = nn(size*2, size)
    __slots__ = '_size', '_nn', '_single', '_left', '_right'

    def __call__(self, x):
        """
        Gets the estimate.
        
        To be used only inside `with _Seen(): ...`!

        If a non-transparent whitebox, associates a directly-adjustable variable with it.
        Else, combines all children representations.

        No child→child nor parent→child dataflow, so results are cached.
        Suitable for structures with high degree of sharing.
        """
        if not isinstance(x, Struct) and not _open_whitebox(x):
            # Variable.
            if id(x) not in self._single:
                self._single[id(x)] = self._nn(None, self._size)
            return self._single[id(x)](None)

        # Cache. Increase ref-count. If it was 0, proceed to the below.
        if _Seen.has(self, x):
            n = _Seen.get(self, x)
            n[1] += 1 ; n[2] += 1
            if _Replay.writing():
                _Replay.add(n[4])
            return n[0]
        initial = np.zeros(self._size); initial[0] = 1.
        _Seen.set(self, x, [initial, 1, 1, None, ...]) # Don't deal with graphs.
        # [result, refcount, max_refcount, dout, ReplaySlice]

        # Combine children.
        sl = _Replay.slice() if _Replay.writing() else None
        children = x if isinstance(x, Struct) else _open_whitebox(x)
        n = len(children)
        with SetExecState(self, ([None]*n, [None]*n, [None]*n, [None]*n)) as state:
            o,l,m,r = state
            for i in range(n):
                o[i] = self(children[i])
                arg = np.concatenate((o[i], l[i-1] if i > 0 else initial))
                l[i], m[i] = np.split(self._left(arg), (self._size,))
            for i in reversed(range(n)):
                arg = np.concatenate((m[i], r[i+1] if i+1 < n else initial))
                r[i] = self._right(arg)
            _Seen.get(self, x)[0] = r[0]
            if sl is not None: _Seen.get(self, x)[4] = _Replay.slice(sl)
            return r[0]

    def adjust(self, ins, out, dout):
        x, = ins
        if not isinstance(x, Struct) and not _open_whitebox(x):
            assert id(x) in self._single
            return adjust(self._single[id(x)], None, out, dout)

        # Uncache. Decrease ref-count. If it becomes 0, proceed to the below.
        assert _Seen.has(self, x)
        n = _Seen.get(self, x)
        dout = n[3] = _merge(n[3], dout)
        n[1] -= 1
        if n[1] > 0: return
        if dout is not None:
            dout /= n[2]
        _Seen.pop(self, x)

        # Adjust the combination of children.
        children = x if isinstance(x, Struct) else _open_whitebox(x)
        n = len(children)
        with GetExecState(self) as state:
            # Rely on AutoFuncs already storing inputs in their state (so we can pass ins=None to `adjust`).
            o,l,m,r = state
            initial0 = np.zeros(self._size)
            dl, dm, dr = initial0, [None]*n, dout
            for i in range(n):
                dins, = self._right.adjust(None, r[i], dr)
                if dout is not None:
                    dm[i], dr = np.split(dins, (self._size,))
            dch = dr = None
            for i in reversed(range(n)):
                if dout is not None:
                    out = np.concatenate((l[i], m[i]))
                    dout = np.concatenate((dl, dm[i]))
                dins, = self._left.adjust(None, out, dout)
                if dout is not None:
                    dch, dl = np.split(dins, (self._size,))
                self.adjust((children[i],), o[i], dch)
            dl = None



class _default_nn:
    def __init__(self, learning_rate = 1, hidden_layers = 1):
        self.lr = learning_rate
        self.hl = hidden_layers
    def __call__(self, in_size, out_size):
        """
        The default building block that can be executed and adjusted to minimize error of prediction.
        Just creates a couple dense and leaky relu layers.
        """
        if in_size is None:
            body = var(np.random.normal(0, 1, out_size), self.lr)
        else:
            hidden_layers, hidden_size = self.hl, out_size+20
            L, sz = args(), in_size
            for _ in range(hidden_layers):
                w = np.random.normal(0, 1/sz, (sz, hidden_size))
                L = Struct(dense, L, var(w, self.lr))
                np.fill_diagonal(w, 1) # Bias output towards identity, not zero.
                L = Struct(leaky_relu, L)
                sz = hidden_size
            w = np.random.normal(0, 1/sz, (sz, out_size))
            np.fill_diagonal(w, 1)
            body = Struct(dense, L, var(w, self.lr))
        return AutoFunc(body = body)



class _Replay:
    """
    Using a replay buffer means to be able to replace every non-static part
    of control flow with its statically-remembered result.

    Everything using randomness or learned state to determine control-flow should be
    `fc(_Replay.get() if _Replay.reading() else _Replay.add(…))` on call,
    `fa(_Replay.get())` on adjust.

    Replaying itself is a non-pure op, so the `_Replay` object or its tape must be saved in replays.

    If executing without a replay or adjustment, make sure to call `_Replay.pause()` afterwards.
    """
    def __init__(self, tape = ..., state = None):
        """
        Pass in nothing to record a replay.
        Enter a replay as a context manager to replay the experience.
        """
        assert tape is ... or isinstance(tape, list)
        self.rc = 0
        self.prev = self.prev_state = None
        self.tape = tape if tape is not ... else []
        self._state = state
        self._reset()
        if len(self.tape) == 0:
            self.direction = _Replay._write
            self.tape.append([]) # For `impure`.

    def __enter__(self):
        if self.rc == 0:
            self.prev, TL._Replay = TL._Replay, self
            s2 = dict(TL._Replay_state) if TL._Replay_state is not None else {}
            if self._state: s2.update(self._state)
            self.prev_state, TL._Replay_state = TL._Replay_state, s2
        self.rc += 1
    def __exit__(self, x,y,z):
        self.rc -= 1
        if self.rc == 0:
            prev = TL._Replay
            if prev is not None:
                if prev.direction == _Replay._write:
                    _Replay.reset()
                    prev = None
            if x is not None: # Reset on error to not cause another error on replay.
                self._reset()
            TL._Replay, self.prev = self.prev, None
            TL._Replay_state, self.prev_state = self.prev_state, None
            # If you don't have the habit of developing slowly and
            #   don't check every part for slightest errors,
            #     and this is quite the nasty assertion.
            if x is None: assert prev is None, "Somewhere before, did not _Replay.get() enough."

    def __getstate__(self):
        return self.tape, self._state
    def __setstate__(self, st):
        self.__init__(*st)

    __slots__ = 'tape', 'direction', 'position', 'impure_position', '_state', 'prev', 'prev_state', 'rc'

    _write = 0
    _read = 1
    _backward = 2

    @staticmethod
    def reading():
        """
        Returns whether we are currently replaying an experience.

        If True, the dynamic part should use `_Replay.get()` to repeat a prior result,
        else `_Replay.set(…)` to be able to repeat it later.

        Do not use this on adjustment.
        """
        self = TL._Replay
        return self is not None and self.direction == _Replay._read
    @staticmethod
    def writing():
        """Returns whether we are writing a replay.

        Do not use this on adjustment."""
        return TL._Replay is not None and TL._Replay.direction == _Replay._write

    @staticmethod
    def impure_get():
        """
        Remembers the previous value on replay. For non-adjustable (user) code.

        if _Replay.reading():
            return _Replay.impure_get()
        else:
            return _Replay.impure_add(func(*args, **kwargs))
        """
        self = TL._Replay
        v = self.tape[0][self.impure_position]
        self.impure_position += 1
        if self.position == 1 and self.impure_position == len(self.tape[0]):
            self._reset()
            TL._Replay = None
        return v

    @staticmethod
    def impure_add(v):
        """
        Remembers the value to replay. For non-adjustable (user) code.
        """
        if TL._Replay is not None:
            TL._Replay.tape[0].append(v)
            TL._Replay.impure_position += 1
        return v
        

    @staticmethod
    def get():
        """
        Returns the dynamically-computed-if-not-replaying value.

        Always use this on adjustment, and conditionally on execution before that.
        """
        self = TL._Replay
        assert self is not None
        if self.direction == _Replay._read and self.position == len(self.tape):
            assert self.impure_position == len(self.tape[0]), "Some impure behavior is not @impure"
        if self.direction == _Replay._write or self.position == len(self.tape):
            assert len(self.tape) > 1, "Nothing to get; wrong replay, or didn't add anything"
            self.direction = _Replay._backward
            self.position = len(self.tape) - 1
        v = self.tape[self.position]
        self.position += 1 if self.direction == _Replay._read else -1
        if self.position == 0 and self.impure_position == len(self.tape[0]):
            self._reset()
            TL._Replay = None
        return v

    @staticmethod
    def add(v):
        """
        Remembers a value for replay.
        """
        self = TL._Replay
        if self is None:
            self = TL._Replay = _Replay()
        assert self.direction == _Replay._write
        if not isinstance(v, _ReplaySlice):
            self.tape.append(v)
        elif id(self) not in v.seen:
            self.tape[0].extend(v.tape[0][v.ip1 : v.ip2])
            self.tape.extend(v.tape[v.p1 : v.p2])
            v.seen.append(id(self))
        return v

    @staticmethod
    def pause():
        """
        Pauses replay at the end of its scope; returns the replay object.
        Use for execution without adjustment, or non-top-level execution+adjustment.
        """
        self = TL._Replay
        TL._Replay = None
        return self

    @staticmethod
    def reset():
        """
        Resets the current replay to its initial ready-to-read state.
        Use when an exception arises near the creation of a replay.
        """
        if TL._Replay is not None:
            self = TL._Replay._reset()
            TL._Replay = None
            return self
    def _reset(self):
        self.direction = _Replay._read
        self.position = 1
        self.impure_position = 0
        return self


    @staticmethod
    def state(keyed, st = ...):
        """Allows remembering state in newly-created replay buffers.
        For functions that will change control flow depending on state."""
        if st is ...:
            "Read TL._Replay_state at `keyed`."
            if TL._Replay_state is None: return None
            return TL._Replay_state.get(keyed, None)
        else:
            "Write: return a ctx manager that sets current threadlocal state at `keyed` to `st`."
            return _ReplayState(keyed, st)

    @staticmethod
    def slice(t = ...):
        """Allows remembering slices of the current replay buffer for possible later insertion into others.
        For functions that implement replay-buffer-aware caching with `_Seen` (such as `_renn`)."""
        self = TL._Replay
        assert self.direction == _Replay._write
        if t is ...:
            return self.impure_position, len(self.tape)
        else:
            return _ReplaySlice(self, t)

class impure:
    """
    A decorator for functions that are non-learnable but potentially-non-deterministic, like network IO.
    Their results are saved separately in replay buffers.
    """
    def __call__(self, /, *args, **kwargs):
        if _Replay.reading():
            return _Replay.impure_get()
        else:
            return _Replay.impure_add(self._f(*args, **kwargs))
    adjust = None

    __slots__ = '_f',
    def __init__(self, f):
        self._f = f
    def __getattr__(self, key):
        return getattr(self._f, key)


class _ReplayState:
    """A context manager for adding to the remembered state of newly-created-inside replay buffers."""
    def __init__(self, k, v):
        self.prev = None
        self.k = k
        self.v = v
    __slots__ = 'prev', 'k', 'v'
    def __enter__(self):
        assert self.prev is None
        prev = TL._Replay_state
        # TL._Replay_state is immutable.
        new = dict(prev) if prev is not None else {}
        new[self.k] = self.v
        self.k, self.v = None, None
        self.prev, TL._Replay_state = prev, new
        return self
    def __exit__(self, x,y,z):
        TL._Replay_state, self.prev = self.prev, None

class _ReplaySlice:
    """A read-only slice of a replay buffer that can be added to another replay buffer."""
    def __init__(self, r, t):
        self.tape = r.tape
        self.ip1, self.p1 = t
        self.ip2, self.p2 = r.impure_position, len(r.tape)
        self.seen = [id(r)] # To not add the same slice to the same replay twice.
    __slots__ = 'tape', 'ip1', 'ip2', 'p1', 'p2', 'seen'



class replay:
    """
    A decorator that makes a function remember recent replay buffers when called, for reliving the experience.

    @replay()
    def f(): ...
    """
    all = weakref.WeakValueDictionary({})
    def __init__(self, limit = 100):
        self.limit = limit
    def __call__(self, func):
        if self.limit == 0: return func
        return _replayed_func([], func, self.limit)

    @staticmethod
    def add(func, ins, replay):
        """
        Adds inputs and replay to a function's replay buffer.
        """
        assert isinstance(func, _replayed_func)
        func._dataset.append((ins, replay))
        if func._limit > 0 and len(func._dataset) >= func._limit * 2: # Ensure linear runtime with `* 2`.
            del func._dataset[:-func._limit]

    @staticmethod
    def redo(what = ..., single_thread = True, work_for = 1):
        """
        Trains a function or a list of functions.

        Parameters
        ==========
        - ``what``: what to train (repeatedly execute+adjust).
        Is `replay.all` by default, as a moonlight guides.
        - ``single_thread``: False to train in parallel.
        - ``work_for``: how many seconds to train for.
        """
        if what is ...:
            what = replay.all

        if isinstance(what, dict) or isinstance(what, weakref.WeakValueDictionary):
            fs = [*what.values()]
        elif isinstance(what, list) or isinstance(what, tuple):
            fs = what
        else:
            fs = (what,)

        import time
        end = time.perf_counter() + work_for
        if single_thread:
            while True:
                try:
                    _redo_all(fs, end)
                except RecursionError:
                    pass
                if end < time.perf_counter(): break
        else:
            # `pickle` tries to deconstruct globals, so
            #   we get pickling errors. I don't know how to fix it
            #     (without imposing a huge burden on the user, with an out-of-line definition).
            parallel(_redo_all, fs, end, work_for = work_for)

def _redo_all(fs, end):
    for f in fs:
        f._redo(end)



class _replayed_func:
    def __call__(self, *ins):
        """
        Adds a new replay to the current replay,
        executes the function,
        and adds function inputs/output/tape to its dataset.

        Even recursive replays will be saved & restored correctly.
        """
        if self._limit == 0:
            return self._func(*ins)
        if _Replay.reading():
            r = _Replay.impure_get()
        else:
            r = _Replay()
        self._visits += 1
        try:
            with r:
                out = self._func(*ins)
                # Pause `r`, but copy its reset version into the replay buffer.
                _Replay.pause()
                return out
        finally:
            self._visits -= 1
            if not _Replay.reading():
                if self._visits == 0: # Ensure only one replay per call tree.
                    replay.add(self, ins, _Replay(r.tape, r._state))
                if _Replay.writing():
                    _Replay.impure_add(r)

    def _redo(self, end):
        """Fits the function to known output & replay, once, at top-level."""
        assert TL._Replay is None, "Not at top-level"
        with ExecState():
            for batch in var.batch(self._dataset):
                for ins, r in batch:
                    if r is None:
                        self._func(*ins)
                    else:
                        with r:
                            self._func(*ins)
                    if end < time.perf_counter():
                        var.commit()
                        return
                var.commit()


    __slots__ = '_dataset', '_func', '_limit', '_visits', '__weakref__'
    def __init__(self, dataset, func, limit):
        assert not hasattr(func, 'adjust')
        self._dataset = dataset
        self._func = func
        self._limit = limit
        self._visits = 0
        replay.all[id(self)] = self

    def __getstate__(self):
        return self._dataset, self._func, self._limit
    def __setstate__(self, state):
        self._dataset, self._func, self._limit = state
    def __getattr__(self, key):
        return getattr(self._func, key)



class var:
    # I do know that re-implementing things that are already better done in other places
    #   is not the brightest idea. But I needed to understand how it's done.
    """A variable that holds and subtracts from a numpy array, for SGD with Nesterov momentum."""
    def __new__(cls, v, speed, deceleration = .9):
        return Struct(cls, var._see(v), var._see(np.zeros_like(v)), speed, deceleration)
    def call(v, momentum, speed, decel):
        return v + decel * momentum
    def adjust(i,o,do):
        if do is None: return
        v, momentum, speed, decel = i
        # Standard SGD is adding `-speed * do` to `v` (decel=0).
        # But just one learning_rate switch is not the perfection.
        # Seeing the code, it can be made into anything: momentum with state like here, a NN, etc.
        mom2 = decel * momentum - (1 - decel) * speed * do
        var.update(momentum, mom2 - momentum)
        var.update(v, mom2)

    def filter(_, __): return False

    @staticmethod
    def commit(cancel = False):
        """Commits changes to numeric variables that were made with `var.update`.

        Parameters
        ==========
        - ``cancel``: If True, discards updates."""
        d = TL.var_deltas
        if not cancel:
            # Share data with other processes, if any.
            if var._puts is not None:
                assert var._gets is not None and len(var._puts) == len(var._gets)
                import pickle
                update = pickle.dumps([(var._to_split[id(v)], delta, n) for v, delta, n in d.values()])
                for q in var._puts:
                    q.put(update, True, .1)
                for q in var._gets:
                    first = True
                    while True:
                        # Read from each queue until an exception (Empty).
                        try: share = q.get(first, .1)
                        except: break
                        first = False
                        for name, delta, n in pickle.loads(share):
                            if name in var._to_join:
                                var.update(var._to_join[name], delta, n)
                            else:
                                var.joining_errors += 1

            # Commit.
            for v, delta, n in d.values():
                # Changing global learning rate can't compensate for this division
                #   if in non-fixed control flow.
                delta /= n

                # Also do some gradient rescaling.
                l = np.sqrt(np.sum(delta * delta))
                if l > 1: delta *= 1 / l

                v[:] = v + delta; v[v == 0] = 1e-10

        d.clear()

    @staticmethod
    def batch(dataset, num_batches = 5):
        """
        Batches a dataset (or an array), distributing work between processes if needed.

        Example of usage:

        for batch in var.batch(dataset):
            for ins, out in batch:
                pred = func(*ins)
                adjust(func, ins, pred, pred - out)
            var.commit()
        """
        I, N = var._process_index, 1 if var._puts is None else 1+len(var._puts)
        L = len(dataset)
        r = range(int(L * I/N), int(L * (I+1)/N))
        indexes = random.sample(r, len(r))
        step = int(1 + (len(r)-1) / num_batches) # Round up.
        for i in range(num_batches):
            x, y = min(i * step, len(r)), min((i+1) * step, len(r))
            yield (dataset[indexes[j]] for j in range(x, y))


    @staticmethod
    def update(v, delta, n = 1):
        """Adds `delta` to `v`, later at `var.commit`.
        Done automatically when a variable receives gradient."""
        # A much more efficient way to do these would be to
        #   store both value and delta and n in one NumPy array.
        #   Almost no dict lookups, then (except at inter-process messaging).
        assert v.shape == delta.shape, "Wrong shapes"
        d, i = TL.var_deltas, id(v)
        if i in d:
            d[i][1] += delta
            d[i][2] += n
        else:
            var._see(v) # Shore up possible users' forgetfulness.
            d[i] = [v, delta, n]



    # De/serialization and parallelization stuff below.

    @staticmethod
    def is_main_process():
        """Returns True if this process was not spawned by `parallel`."""
        return var._process_index == 0

    joining_errors = 0

    _puts = None
    _gets = None
    _process_index = 0

    _free_names = [0]
    _to_join = weakref.WeakValueDictionary() # name → array
    _to_split = {} # id(array) → name
    @staticmethod
    def _see(arr, name = None):
        """Remembers the (NumPy) array for splitting/joining.
        Call this when creating the array that will be passed to `var.update(array, …)`."""
        if id(arr) in var._to_split: return arr
        names = var._free_names
        if name is None:
            name = names.pop() if len(names) > 1 else names[0]
            if len(names) == 1: names[0] += 1
        elif name+1 > names[0]:
            names[0] = name+1

        if name in var._to_join and not np.array_equal(arr, var._to_join[name]) and not np.isnan(np.sum(arr)):
            print('had', var._to_join[name])
            print('got', arr)
            assert False, f"Duplicate name {name} for a newly-seen array"

        var._to_split[id(arr)] = name
        var._to_join[name] = arr
        weakref.finalize(arr, var._unsee, name)
        return arr
    @staticmethod
    def _unsee(name):
        if name in var._to_join:
            i = id(var._to_join[name])
            if i in var._to_split:
                del var._to_split[i]
            del var._to_join[name]
            var._free_names.append(name)
    @staticmethod
    def _split(obj):
        """
        Pickling.
        Splits an object for giving from a parent process to its child.
        The batch-parallelizing part of `var.commit` will be available.
        """
        import pickle
        return pickle.dumps((obj, dict(var._to_join)), protocol=-1)
    @staticmethod
    def _join(pickled):
        """
        Unpickling.
        Accepts an object given to a child.
        """
        import pickle
        obj, tj = pickle.loads(pickled)
        for name, array in tj.items():
            var._see(array, name)
        return obj



def parallel(func, *args, processes = ..., work_for = 10, warmup = True):
    """
    Calls func with args many times.

    Call this inside `if __name__ == '__main__': ...` to
    not spawn processes endlessly. (Unneeded if forking.)

    `var.commit()` inside will share data with other processes.
    `for batch in var.batch(dataset): ...` will split the workload evenly.

    Keyword parameters
    ==================
    - ``processes``: how many processes to fork/spawn.
    By default, is `multiprocessing.cpu_count`.
    - ``work_for``: how many seconds this call is guaranteed to last.
    10 by default.
    - ``warmup``: whether we need one function call in the main thread to warm up data sharing.
    (We do, with the current implementation of `_renn` and such.)
    """
    if warmup: func(*args) # We fill id(obj)→var dicts dynamically, so we need to warm up.

    # Have a queue (of variables' updates) from each process to each process.
    import multiprocessing as mp
    N = mp.cpu_count() if processes is ... else processes
    if N == 1:
        return _accept_queues(work_for, 0, None, None, (func, args))

    q = [[mp.Queue() if i != j else None for j in range(N)] for i in range(N)]
    puts = [[q[i][j] for j in range(N) if i != j] for i in range(N)]
    gets = [[q[j][i] for j in range(N) if i != j] for i in range(N)]
    fas = var._split((func, *args))
    ps = [mp.Process(target = _accept_queues, args = (work_for, i, puts[i], gets[i], fas)) for i in range(1, N)]
    for p in ps: p.start()
    try:
        _accept_queues(work_for, 0, puts[0], gets[0], (func, args))
        for p in ps: p.join()
    except:
        for p in ps: p.kill()
        raise

def _accept_queues(work_for, ind, puts, gets, fas):
    if ind != 0:
        func, *args = var._join(fas)
    else:
        func, args = fas
    import time
    assert var._puts is None and var._gets is None, "Call to `parallel` inside `parallel`."
    var._puts, var._gets, var._process_index = puts, gets, ind
    end = time.perf_counter() + work_for
    try:
        while True:
            func(*args)
            if end < time.perf_counter(): break
    finally:
        var._puts, var._gets, var._process_index = None, None, 0



_persist_finalizer = None
class persist:
    """
    Decorator for functions the attributes of which can be saved/loaded.
    (Only the main process can save/load anything.)
    """
    def __new__(cls, f):
        assert callable(f) and hasattr(f, '__code__')
        persist.all[id(f)] = f
        persist._load(f)
        return f
    all = weakref.WeakValueDictionary()
    loading = {}

    @staticmethod
    def save(path = 'state.bin'):
        """
        Saves all attributes of all persisted functions to a file.
        """
        if not var.is_main_process(): return
        if len(persist.all) == 0: return
        full = [(*persist._clues(f), persist._getattr(f)) for f in persist.all.values()]

        import pickle
        with open(path, mode = 'wb') as file:
            pickle.dump(full, file, protocol=-1)
    @staticmethod
    def load(path = 'state.bin'):
        """
        Loads all attributes of all persisted functions (both before and after this) from a file.
        """
        if not var.is_main_process(): return
        import pickle
        try:
            with open(path, mode = 'rb') as file:
                full = pickle.load(file)
        except OSError:
            return

        persist.loading = {}
        for *clues, attrs in full:
            for c in clues:
                persist.loading[c] = attrs
        for f in persist.all.values():
            persist._load(f)
    @staticmethod
    def _clues(f):
        """Returns strings one of which should match (preferably the first one) for
        a loaded-attr-dict to be considered the loading-from target for a function."""
        return '::' + f.__code__.co_filename + '::' + f.__code__.co_name,   f.__code__.co_code
    @staticmethod
    def _getattr(f):
        """Attributes of a function."""
        if hasattr(f, '__getstate__') and hasattr(f, '__setstate__'):
            return f.__getstate__()
        if hasattr(f, '__slots__'):
            if hasattr(f, '__dict__'):
                return (*f.__slots__, f.__dict__)
            return (*f.__slots__,)
        return f.__dict__
    @staticmethod
    def _setattr(f, to):
        """Sets attributes of a function."""
        if hasattr(f, '__getstate__') and hasattr(f, '__setstate__'):
            f.__setstate__(to)
        elif hasattr(f, '__slots__'):
            if hasattr(f, '__dict__'):
                assert len(f.__slots__) == len(to)-1
                for i, k in enumerate(f.__slots__):
                    # Python insidiously forces *so many* inefficiencies on programmers.
                    #   (I noticed many itty-bitty instances of this before, too.)
                    setattr(f, k, to[i])
                f.__dict__.update(to[-1])
            else:
                assert len(f.__slots__) == len(to)
                for i, k in enumerate(f.__slots__):
                    setattr(f, k, to[i])
        else:
            assert isinstance(to, dict)
            f.__dict__.update(to)
    @staticmethod
    def _load(f):
        """Fills attributes of a function if present."""
        if not var.is_main_process(): return
        clues = persist._clues(f)
        for c in clues:
            if c in persist.loading:
                persist._setattr(f, persist.loading[c])
                del persist.loading[c]
        # Else, `f` is created. (Or not overriden by loading, at least.)



# Perfectly extensible (via the "modify the library code" principle).
#   (Don't see a way to get convenience AND customizability in Python.)
persist.load()
_persist_finalizer = type('_persist_finalized', (), {})()
weakref.finalize(_persist_finalizer, persist.save)



class _deferred_result:
    """Allows lazy computation of the deferred result.
    Holds a function that will be called when it's time to adjust."""
    def __init__(self, f): self.f = f
    __slots__ = 'f',

class scaffolding:
    """
    `scaffolding(…)`: Decorators that embed differentiable scaffolding into arbitrary code.

    Non-deterministic decisions should all be replaced with ones that depend on this.
    Vocabulary: "random" → "arbitrary".

    Usage example (the written `x` will be maximized by choices):

    @cached
    @replay()
    @static()
    def f(n):
        scaffolding.see(n)
        return state.x(0,1)

    ctx = static(x = choice())
    with ctx:
        state.x = good_of(f(random.randrange(5)))
    """

    deferred = _deferred_result

    def __init__(self, feature_size, nn = ..., value_estimator_creator = _renn, loss = loss2, auto = False):
        """
        Takes representation size, adjustable NN building block creator, and
        (a creator of) a adjustable function from arbitrary-value to representation.

        Parameters
        ==========
        - ``feature_size``: size of the hidden representation.
        Multiples of this are passed to ``nn``.
        - ``nn``: creator of basic neural network units (from input and output sizes).
        The result can be called with one arg (a NumPy array) then `adjust`ed.
        Is `_default_nn(.1)` by default.
        - ``value_estimator_creator``: creates the recursive NN when passed feature size and nn.
        - ``loss``: the adjustable loss that takes prediction and ideal and
        adjusts prediction to have the gradient of, say, prediction minus ideal.
        - ``auto``: True to call `persist` and `replay()` on all functions
        decorated with this scaffolding.
        """
        if nn is ...:
            nn = _default_nn(.1)
        self._size = feature_size
        self._nn = nn
        self._value_creator = value_estimator_creator
        c = self._cur = threading.local()
        c.repr = None
        c.node = None
        c.seen = None
        c.exposed = None
        self._loss = loss
        self._rnn = nn(feature_size*2, feature_size)
        self._value = self._value_creator(self._size, self._nn)
        self._auto = auto
    __slots__ = '_loss', '_rnn', '_value', '_cur', '_size', '_nn', '_value_creator', '_auto', '__weakref__'
    def __call__(__self, /, **exposed_types):
        """
        Accepts what state a scaffolding's node will expose.
        """
        for k in exposed_types:
            # Recognize [num(), num()], {'k':num()}, {0,1,2}, bool, range(3).
            exposed_types[k] = _exposed_type_to_convenience(exposed_types[k])
        return _scaffolding_context(__self, exposed_types)

    def __getstate__(self):
        return self._loss, self._rnn, self._value, self._size, self._nn, self._value_creator, self._auto
    def __setstate__(self, st):
        self._loss, self._rnn, self._value, self._size, self._nn, self._value_creator, self._auto = st
        c = self._cur = threading.local()
        c.repr = None
        c.node = None
        c.seen = None
        c.exposed = None

    def get(self):
        """
        Returns the current hidden state.
        """
        return self._cur.repr
    @staticmethod
    def see(x, add_repr = None):
        """
        Mixes the value's representation into the current hidden state.
        """
        self = TL.cur_scaffold
        prev_outer_node = _get_outer_node(); _set_outer_node(self)
        try:
            c = self._cur
            prev_repr = c.repr
            if x is not scaffolding._takes_from_parent and x is not scaffolding._gives_to_parent:
                add_repr = self._value(x)
            c.repr = self._rnn(np.concatenate((c.repr, add_repr)))
            c.seen.append(('see', prev_repr, x, add_repr))
        finally:
            _set_outer_node(prev_outer_node)
    def seen(func, ins, out):
        """
        Like `.see`, but later adjusts a user-defined computation
        (which returned a 1D NumPy array of an appropriate size).
        """
        self = TL.cur_scaffold
        assert len(out.shape) == 1, "Wrong shape"
        assert out.size == self._size, "Wrong size"
        prev_outer_node = _get_outer_node(); _set_outer_node(self)
        try:
            c = self._cur
            prev_repr = c.repr
            c.repr = self._rnn(np.concatenate((c.repr, out)))
            c.seen.append(('seen', prev_repr, func, ins, out))
        finally:
            _set_outer_node(prev_outer_node)
        
        # Having `.seen(func, ins, out)` to mix adjustable computations could be convenient too.

    _gives_to_parent = object()
    _takes_from_parent = object()

class _scaffolding_state:
    """Exposes the current predictions to functions."""

    def __getattr__(self, key):
        """Returns the prediction from local info of the value at `key`."""
        sc = TL.cur_scaffold
        prev_outer_node = _get_outer_node(); _set_outer_node(sc)
        try:
            c = sc._cur
            f = c.exposed[key]
            value = f(c.repr)
            c.seen.append(('get', key, value, not hasattr(f, 'calc_loss_ourselves')))
            return value
        finally:
            _set_outer_node(prev_outer_node)
    def __setattr__(self, key, value):
        """Later backpropagates (the error of not being) the actual value at `key`.

        (Also see `scaffolding.deferred`.)"""
        c = TL.cur_scaffold._cur
        assert key in c.exposed
        c.seen.append(('set', key, value))
    def __delattr__(self, key):
        """Regenerates the method of prediction of the value at `key`."""
        c = TL.cur_scaffold._cur
        c.node._gen(key)
state = _scaffolding_state()




class _scaffolding_context:
    """
    Holds what state a scaffolding's node will expose.
    Can use instances of this as context managers at the top level of execution,
    before calling functions with scaffolding.
    """
    def __init__(self, scaf, exposed_types):
        self._scaffold = scaf
        if len(exposed_types):
            self._exposed = {
                k: T(scaf._size, scaf._nn, self)
                for k,T in exposed_types.items()
            }
            self._exposed_types = exposed_types
        else:
            self._exposed = None
            self._exposed_types = None
    __slots__ = '_scaffold', '_exposed', '_exposed_types', '_prev_scaffold'
    def __call__(self, func):
        """
        Decorates a function with scaffolding.
        Also auto-persists every function.
        """
        out = _node(func, self._scaffold, self._exposed, self._exposed_types)
        if self._scaffold._auto:
            out = replay()(out)
            out = persist(out)
        return out

    def __enter__(self):
        c = self._scaffold._cur
        assert c.node is None and c.exposed is None, "Not top-level"
        c.repr = np.zeros(self._scaffold._size)
        c.repr[0] = 1.
        c.node = self
        c.seen = []
        if _Replay.state(self._scaffold) is not None:
            c.exposed = _Replay.state(self._scaffold)[0]
        else:
            c.exposed = dict(self._exposed) if self._exposed is not None else {}
        c.writes = None
        self._prev_scaffold, TL.cur_scaffold = TL.cur_scaffold, self._scaffold

        es = ExecState()
        r = None if TL._Replay is not None else _Replay()
        s = None if _Seen.exists() else _Seen()
        TL.scaffolding = es,r,s
        es.__enter__(), r and r.__enter__(), s and s.__enter__()
    def __exit__(self, x,y,z):
        c = self._scaffold._cur
        try:
            c.writes = None
            if x is None:
                if _Replay.state(self._scaffold) is not None:
                    _node._adjust(c.repr, np.zeros_like(c.repr), c.node, c.seen, c.exposed)
                else:
                    with _Replay.state(self._scaffold, (c.exposed, {})):
                        _node._adjust(c.repr, np.zeros_like(c.repr), c.node, c.seen, c.exposed)
            es,r,s = TL.scaffolding
            es.__exit__(x,y,z), r and r.__exit__(x,y,z), s and s.__exit__(x,y,z)
        finally:
            TL.cur_scaffold, self._prev_scaffold = self._prev_scaffold, None
            c.repr = c.node = c.seen = c.exposed = None
            if TL.cur_scaffold is None:
                TL._Replay = None # A band-aid for us not resetting replay in *some* circumstances.

class _node:
    """
    A wrapped function that's a part of differentiable scaffolding.
    """
    def __init__(self, func, scaffold, exposed, exposed_types):
        """Generates the exposed interface."""
        self._func = func
        self._scaffold = scaffold
        self._exposed = exposed
        self._exposed_types = exposed_types
        self._prev_scaffold = None
    __slots__ = '_func', '_scaffold', '_exposed', '_exposed_types', '_prev_scaffold'

    def __getattr__(self, key):
        return getattr(self._func, key)

    def __getstate__(self):
        return self._func, self._scaffold, self._exposed, self._exposed_types
    def __setstate__(self, st):
        self.__init__(*st)

    def _gen(self, key):
        """Re/generates the exposed predictor at `key`."""
        if self._exposed is None: return
        sc = self._scaffold
        self._exposed[key] = self._exposed_types[key](sc._size, sc._nn, self)

    def __call__(self, /, *args, **kwargs):
        """Handles scaffolding-keeping and calls the user's function."""
        sc = self._scaffold
        c = sc._cur
        was_top_level = False
        if c.node is None: # If top-level, set things up.
            # (Note: this is just a convenience for simple cases.)
            #   (Strongly prefer explicit `ctx = static(a=num()) ; with ctx: ...`.)
            was_top_level = True
            _scaffolding_context.__enter__(self)
        # Temporarily replace parent's current stuff with our own.
        prev_scaffold, TL.cur_scaffold = TL.cur_scaffold, sc
        prev = c.seen, c.node, c.repr
        repr = c.repr = np.zeros(sc._size)
        repr[0] = 1.
        c.node = self
        seen = c.seen = []

        if _Replay.reading():
            prev_exposed, c.exposed = c.exposed, _Replay.state(sc)[0]
            state = None
        elif _Replay.writing():
            # Recalc `exposed` and remember if our first time.
            if self._exposed is not None:
                prev_exposed, c.exposed = c.exposed, dict(c.exposed)
                c.exposed.update(self._exposed)
            else:
                prev_exposed = None
            state = _Replay.state(sc, (c.exposed, {})).__enter__()
        else:
            assert False, "Tried to read from replay during call"

        prev_outer_node = _get_outer_node(); _set_outer_node(sc)
        try:
            # Set up the initial repr. Depend on parent and self.
            if prev[2] is not None: scaffolding.see(scaffolding._takes_from_parent, prev[2])
            scaffolding.see(self)

            # Do the actual non-scaffolding work.
            return self._func(*args, **kwargs)
        except:
            sc = None
            raise
        finally:
            repr = c.repr
            _set_outer_node(prev_outer_node)
            c.seen, c.node, c.repr = prev
            if prev_exposed is not None:
                c.exposed = prev_exposed
            c.seen.append(('call', repr, self, seen, TL._Replay))
            scaffolding.see(scaffolding._gives_to_parent, repr)
            TL.cur_scaffold = prev_scaffold
            if state is not None:
                if TL._Replay._state is None:
                    # Ensure that @replay() @static() will work.
                    TL._Replay._state = TL._Replay_state
                state.__exit__(None if sc is not None else True, None, None)
            if was_top_level:
                _scaffolding_context.__exit__(self, None if sc is not None else True, None, None)

    @staticmethod
    def _adjust(repr, drepr, node, seen, exposed):
        """
        The top-level special adjustment of scaffolding.
        Reverses __call__. Returns gradient of parent's repr.
        """
        # Mix in exposed predictors.
        if node._exposed is not None:
            exposed = dict(exposed) ; exposed.update(node._exposed)

        sc = node._scaffold
        exposed_got = sc._cur.writes

        # Recall sc._cur.writes (and remember writes-to-exposed if our first-for-scaffold time).
        if TL._Replay is not None:
            if sc._cur.writes is None:
                exposed_got = sc._cur.writes = dict(_Replay.state(sc)[1])
            elif len(_Replay.state(sc)[1]) == 0:
                _Replay.state(sc)[1].update(sc._cur.writes)
        else:
            assert False

        parent_drepr = None
        prev_outer_node = _get_outer_node()
        try:
            for t in reversed(seen):
                _set_outer_node(sc)
                if t[0] == 'get': # Adjust.
                    _, key, value, calc_loss = t
                    if key not in exposed_got:
                        assert False, f"A scaffolding read without a write afterwards at {key}"
                    loss_func = sc._loss
                    ideal = exposed_got[key]
                    if isinstance(ideal, _deferred_result):
                        ideal = ideal()
                    if calc_loss:
                        loss = loss_func(value, ideal)
                        dloss = loss_func.adjust((value, ideal), loss, 0)[0]
                        value, = exposed[key].adjust((repr,), value, dloss)
                    else:
                        value, = exposed[key].adjust((repr,), value, ideal)
                    # Divide by 2 to make var-change-to-output smoother in different control flows.
                    drepr = (drepr + value) / 2
                elif t[0] == 'set': # Remember what to adjust towards.
                    _, key, value = t
                    exposed_got[key] = value
                elif t[0] == 'see': # Adjust an RNN step.
                    # Adjust `repr = sc._rnn(np.concatenate((prev_repr, add_repr)))`.
                    _, prev_repr, x, add_repr = t
                    concat = np.concatenate((prev_repr, add_repr))
                    dconcat, = sc._rnn.adjust((concat,), repr, drepr)
                    dprev, dadd = np.split(dconcat, (sc._size,))
                    repr, drepr = prev_repr, dprev
                    if x is scaffolding._takes_from_parent:
                        # Give to parent. (Taking its repr only happens once, but we're generic here.)
                        parent_drepr = _merge(parent_drepr, dadd)
                    elif x is scaffolding._gives_to_parent:
                        # Give to child. (`dadd` will be used on the previous `_node._adjust`.)
                        pass
                    else:
                        # Adjust `add_repr = sc._value(x)`.
                        sc._value.adjust((x,), add_repr, dadd)
                elif t[0] == 'seen': # Adjust an RNN step of a user's computation.
                    # Adjust `repr = sc._rnn(np.concatenate((prev_repr, out = func(*ins))))`.
                    _, prev_repr, func, ins, out = t
                    concat = np.concatenate((prev_repr, out))
                    dconcat, = sc._rnn.adjust((concat,), repr, drepr)
                    dprev, dout = np.split(dconcat, (sc._size,))
                    repr, drepr = prev_repr, dprev
                    func.adjust(ins, out, dout)
                elif t[0] == 'call': # Recurse.
                    _, final_repr, rec_node, rec_seen, replay = t
                    with replay:
                        drepr += _node._adjust(final_repr, dadd, rec_node, rec_seen, exposed)
                    drepr /= 2
                else:
                    assert False
            return parent_drepr
        finally:
            _set_outer_node(prev_outer_node)

static = scaffolding(30, _default_nn(.1), _renn, loss2)



# Numeric outputs.

class num:
    """
    `num(*sizes)`: when used in a function's scaffolding at key K,
    getting state at K returns a numeric repr using the current accumulated repr.
    Set state at K to adjust the prediction.
    """
    def __init__(self, *sizes):
        self.szs = sizes
    __slots__ = 'szs',
    def __call__(self, size, nn, node):
        """Return a class that will actually compute the num when called."""
        if len(self.szs) == 0:
            return _num_one(nn(size, 1))
        elif len(self.szs) == 1:
            return nn(size, self.szs[0])
        else:
            return _num_tensor(nn(size, _prod(self.szs)), self.szs, _prod(self.szs))

class _num_one:
    """Result of num(): returns exactly one non-NumPy number."""
    def __init__(self, f):
        self.f = f
    __slots__ = 'f',
    def __call__(self, x):
        return self.f(x)[0]
    def adjust(self, ins, out, dout):
        out = np.array((out,), dtype=np.float64)
        if dout is not None:
            dout = np.array((dout,), dtype=np.float64)
        return adjust(self.f, (ins[0],), out, dout)

class _num_tensor:
    """Result of num(A, B, …): reshapes output into a tensor."""
    def __init__(self, f, szs, sz):
        self.f = f
        self.sz = sz
        self.szs = szs
    __slots__ = 'f', 'szs', 'sz'
    def __call__(self, x):
        return np.reshape(self.f(x), self.szs)
    def adjust(self, ins, out, dout):
        return adjust(self.f, (ins[0],), np.reshape(out, self.sz), np.reshape(dout, self.sz))


def _prod(arr, start = 0):
    """Returns the product of items in an iterable."""
    p = 1
    for v in arr:
        if not start: p *= v
        else: start -= 1
    return p



# Choices.

def _default_argmax(a):
    """Sample from offset & clipped softmax of the input array."""
    # I can't even pin down one perfect implementation of rewards→best.
    #   I only need to know: what's a good-enough bootstrapper for learning such things?
    #     Can you imagine one scaffolding choosing how to do choices of another?
    #     O old ones, o great ones, grant us insight.

    # Explore if uncertain, exploit if certain.
    if random.random() < .01:
        # Also explore randomly, whatever.
        return random.randrange(len(a))
    x = np.array(a)
    l, r = np.min(x), np.max(x)
    x -= (l + r) / 2
    r = np.max(x)
    ex = np.exp(x - r)
    ex[(x < .075) & (x < r)] = 0
    p = random.random() * np.sum(ex)
    for i in range(len(a)):
        p -= ex[i]
        if p < 0:
            return i
    return len(a)-1
#     return max(range(len(a)), key = lambda i: a[i]) # Greedy argmax.


class choice:
    """
    `choice(argmax = ...)(*options)`: when read from a function's scaffolding at key K,
    pass in a list of options to get the (predicted) best option.
    """
    def __init__(self, argmax = _default_argmax):
        self.argmax = argmax
    __slots__ = 'argmax',
    def __call__(self, size, nn, node):
        return _choice_getter(size, nn, self.argmax, node._scaffold)

class _choice_getter:
    """Stage one of a two-stage reward predictor."""
    def __call__(self, r):
        """
        Returns the function to be called, which would return the picked option.
        It MUST be called once and immediately, with possible options as the arguments.
        (So, access as `state.ch(1, 2, 3)` or `state.ch.list([1, 2, 3])`.)
        """
        self._picker._repr = r
        return self._picker
    calc_loss_ourselves = True
    def adjust(self, ins, out, ideal):
        self._picker._repr = ins[0]
        # Counting on having been called.
        return self._picker.adjust(None, None, ideal)

    def __init__(self, size, nn, argmax, scaffold):
        """Get our own option→repr and (repr,repr)→reward differentiable estimators."""
        v = scaffold._value_creator(size, nn)
        self._picker = _choice_picker(nn(size*2, 1), size, v, argmax, scaffold._loss)
    __slots__ = '_picker',


class _choice_picker:
    """Stage two of a two-stage reward predictor. Given options, returns the best option."""
    def __call__(self, *options):
        """Picks the best option."""
        return self.tuple(options)

    def tuple(self, options):
        """Picks the best option from an immutable tuple.

        Exists to not spread options across many args like __call__ would do."""
        return self.pick_and_explain(options)[0]

    def pick_and_explain(self, options):
        """Returns a tuple of the picked option and the list of predictions."""
        assert not isinstance(options, list)
        if len(options) == 1: return options[0]
        r = self._repr
        with SetExecState(self, (options, [None]*len(options), [None]*len(options))) as state:
            _, ests, predictions = state
            for k,o in enumerate(options):
                ests[k] = self._option(o)
                predictions[k] = self._value(np.concatenate((r, ests[k])))[0]
            if _Replay.reading():
                i = _Replay.get()
            else:
                i = _Replay.add(self._argmax(predictions))
            return options[i], predictions

    def iter(self, options):
        """Like `.list`, but copies the option list. Suitable for non-constant lists."""
        return self.tuple((*options,))

    calc_loss_ourselves = True
    def adjust(self, _, __, ideal):
        """Adjusts prediction of reward of the picked option."""
        i = _Replay.get()
        r, dr = self._repr, None
        with GetExecState(self) as state:
            options, ests, predictions = state
            for j,o in enumerate(reversed(options)):
                k = len(options) - j - 1
                if i == k:
                    inp = np.concatenate((r, ests[k]))
                    loss = self._loss_f(predictions[k], ideal)
                    dout, _ = self._loss_f.adjust((predictions[k], ideal), loss, 0)
                    dv, = self._value.adjust((inp,), np.array((predictions[k],)), np.array((dout,)))
                    dr, dest = np.split(dv, (self._size,))
                    self._option.adjust((o,), ests[k], dest)
                else:
                    self._value.adjust(None, predictions[k], None)
                    self._option.adjust((o,), ests[k], None)
            if dr is None:
                print('Expected to have chosen', i, 'but got options', options)
                print('tape:', TL._Replay.tape)
                print('state:', TL._Replay._state)
                assert False, "Internal error"
            return dr,

    def __init__(self, f, size, option, argmax, loss_f):
        self._repr = None
        self._value = f
        self._size = size
        self._option = option
        self._argmax = argmax
        self._loss_f = loss_f
    __slots__ = '_repr', '_value', '_size', '_option', '_argmax', '_loss_f'



# Convenience.

def _exposed_type_to_convenience(t):
    if isinstance(t, list):
        return _exposed_list(t)
    elif isinstance(t, dict):
        return _exposed_dict(t)
    elif isinstance(t, set):
        return static_choice(*t)
    elif t is bool:
        return static_choice(False, True)
    elif isinstance(t, range):
        return static_choice(*(i for i in t))
    else:
        return t




class _exposed_list:
    def __call__(self, size, nn, node):
        L = node._scaffold._loss
        return _exposed_list_creator([_exposed_type_to_convenience(t)(size, nn, node) for t in self.t], L)
    def __init__(self, t):
        self.t = t
    __slots__ = 't',
class _exposed_list_creator:
    def __call__(self, r):
        return [f(r) for f in self.fs]
    calc_loss_ourselves = True
    def adjust(self, ins, out, ideal):
        dr = np.zeros_like(ins[0])
        i = len(self.fs)
        for f in reversed(self.fs):
            i -= 1 # Python's `reversed` is too dumb, can't handle `enumerate`.
            if ideal[i] is not None:
                dL = self.loss.adjust((out[i], ideal[i]), self.loss(out[i], ideal[i]), 0)[0]
                dr += f.adjust(ins, out[i], dL)[0]
            else:
                f.adjust(ins, out[i], None)
        return dr,
    def __init__(self, fs, loss):
        self.fs = fs
        self.loss = loss
    __slots__ = 'fs', 'loss'

class _exposed_dict:
    def __call__(self, size, nn, node):
        z = {k:_exposed_type_to_convenience(t)(size, nn, node) for k,t in self.t.items()}
        return _exposed_dict_creator(z, node._scaffold._loss)
    def __init__(self, t):
        self.t = t
    __slots__ = 't',
class _exposed_dict_creator:
    def __call__(self, r):
        return {k:f(r) for k,f in self.fs.items()}
    calc_loss_ourselves = True
    def adjust(self, ins, out, ideal):
        dr = np.zeros_like(ins[0])
        for k,f in reversed(self.fs.items()):
            if ideal[k] is not None:
                dL = self.loss.adjust((out[k], ideal[k]), self.loss(out[k], ideal[k]), 0)[0]
                dr += f.adjust(ins, out[k], dL)[0]
            else:
                f.adjust(ins, out[k], None)
        return dr,
    def __init__(self, fs, loss):
        self.fs = fs
        self.loss = loss
    __slots__ = 'fs', 'loss'





class static_choice:
    """
    Like `choice(argmax=...)(*options)`, but `static_choice(*options, argmax=...)`.
    """
    def __init__(self, *options, argmax = _default_argmax):
        self.options = options
        self.argmax = argmax
    __slots__ = 'options', 'argmax'
    def __call__(self, size, nn, node):
        return _static_choice_picker(size, nn, self.options, self.argmax, node._scaffold)

class _static_choice_picker:
    """Returns the best option."""
    def __call__(self, r):
        """Returns the picked best option."""
        self._picker._repr = r
        return self._picker(*self._options)
    calc_loss_ourselves = True
    def adjust(self, ins, out, ideal):
        self._picker._repr = ins[0]
        return self._picker.adjust(None, None, ideal)

    def __init__(self, size, nn, options, argmax, scaffold):
        """Get our own option→repr and (repr,repr)→reward differentiable estimators."""
        self._options = options
        v = scaffold._value_creator(size, nn)
        self._picker = _choice_picker(nn(size*2, 1), size, v, argmax, scaffold._loss)
    __slots__ = '_picker', '_options'



class cached:
    """
    A (replay-buffer-aware) decorator for functions that
    are cached for the same first input, like DAG walkers.

    To be used only inside `with _Seen(): ...`, or scaffolding.
    """

    def __init__(self, f, initial = None):
        self.f = f
        self.initial = initial
    __slots__ = 'f', 'initial'

    def __call__(self, *xs):
        # Cache. Increase ref-count. If it was 0, proceed to the below.
        x = xs[0]
        if _Seen.has(self, x):
            n = _Seen.get(self, x)
            n[1] += 1 ; n[2] += 1
            if _Replay.writing():
                _Replay.add(n[4])
            return n[0]
        _Seen.set(self, x, [self.initial, 1, 1, None, ...]) # Don't deal with graphs.
        # [result, refcount, max_refcount, dout, ReplaySlice]

        sl = _Replay.slice() if _Replay.writing() else None
        try:
            out = self.f(*xs)
            _Seen.get(self, x)[0] = out
            return out
        finally:
            if sl is not None:
                _Seen.get(self, x)[4] = _Replay.slice(sl)

    def adjust(self, ins, out, dout):
        x, = ins
        if not isinstance(x, Struct) and not _open_whitebox(x):
            assert id(x) in self._single
            return adjust(self._single[id(x)], None, out, dout)

        # Uncache. Decrease ref-count. If it becomes 0, proceed to the below.
        assert _Seen.has(self, x)
        n = _Seen.get(self, x)
        dout = n[3] = _merge(n[3], dout)
        n[1] -= 1
        if n[1] > 0: return
        dout /= n[2]
        _Seen.pop(self, x)

        return adjust(self.f, ins, out, dout)
