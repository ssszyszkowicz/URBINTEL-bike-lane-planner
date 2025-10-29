# QLANG:
# ><, moves stack
# : rename
# ? dict.get(*,None)
# ! call - not working

# TO DO: qmelt/qunmelt have two definitions!

# https://www.programiz.com/python-programming/operator-overloading
# Empty dictionary (and class?) crashes Q()

from inspect import isclass
from utils import *
import re
from keyword import iskeyword as keyword_iskeyword






if 0:
    print("Q is the best - you will all be assimilated into Q.")
    print("In America, you call functions in Python. In Soviet Python, Q calls YOU!")
    print("In Soviet XYZ, hackers type faster than YOU!")

def Qdisp(obj):
    for p in dir(obj):
        if not p.startswith('__'):
            print(p,'=',Qstr(getattr(obj,p)))



class Qlist(list): pass
def Qmelt(obj):
    if hasattr(obj,'tolist'):
        l = Qlist(obj.tolist())
        l.dtype = obj.dtype
        return l
    if hasattr(obj, 'tolol'):
        l = Qlist(obj.tolol())
        l.dtype = obj.dtype
        return l

def Qunmelt(obj):
    d = get_depth(obj)
    if d == 2:
        return LoL(obj, obj.dtype)
    if d == 1:
        return np_array(obj, obj.dtype) 


def Qe(iterable):
    return enumerate(iterable)

def Qfilter(iterable, condition):
    a,b = len(iterable), len(condition)
    if a!=b: Qer('Qfilter length mismatch:',a,b)
    if Qtype(iterable) == 'numpy.ndarray':
        return iterable[Qbool(condition)]
    return [n for i,n in enumerate(iterable) if condition[i]]


def Qhelp(_locals):
    print('Q is here for YOU!')
    for k in _locals:
        if k.startswith('Q'):
            O = _locals[k]
            if Qtype(O) == 'function': print(Qstr(O))
    print("We don't do no pretty code 'round here. In Soviet XYZ, PEP8 lints YOU!")


def QcFs(func, params=[]):
    return QcF(func, params, silent = True)
def QcF(func, params=[], silent = False):
    p = Qtype(params)
    if (Qtype(func) in Qmx('function method') or hasattr(func,'__call__')) \
       and p in Qmx('tuple list dict'):
        if not silent: print('Calling',Qstr(func),' with','['+', '.join(Qstr(p,fast=True) for p in params)+']...')
        t0 = time_time()
        out = func(**params) if p=='dict' else func(*params)
        t = time_time()-t0
        if not silent: print('...',Qstr(func),'done in %.2f'%t,'seconds.')
        return out
    Qer('Bad QcF call!')

def QeN(iterable, function_v_n, counter = 0):
    t0 = time_time()
    AC = function_v_n.__code__.co_argcount
    out = []
    for n,v in enumerate(iterable):
        if counter and n%counter==0:
            print(str(n)+'/'+str(len(iterable))+' @'+function_v_n.__name__)
        if AC==1: out.append(function_v_n(v))
        elif AC==2: out.append(function_v_n(v,n))
        else: Qer("QeN:function_v_n must take 1 or 2 arguments.")
    t = time_time()-t0
    print('%.2f'%t,'seconds; per call: %.2f ms'%(1000*t/len(iterable)))
    return out





# Add Lol class - L.tolol()

def Qgetallnumpy(obj):
    out = []
    for na in dir(obj):
        if Qtype(getattr(obj, na)) == 'numpy.ndarray':
            out.append(na)
    return out

def Qcont(obj):
    for k in dir(obj):
        if not k.startswith('__'):
            print('.'+k+':',Qstr(getattr(obj,k)))

def Qmem(obj,text=''):
    M = 0
    if Qtype(obj)=='numpy.ndarray': return obj.nbytes
    elif Qtype(obj)=='dict':
        for k in obj:
            if not k.startswith('__'):
                A = obj[k]
                m = Qmem(A,text+'  ')
                M += m
                print(text+'`'+k+' has', int(0.5+m/(2**20)),'MB')
    else:
        for k in dir(obj):
            if not k.startswith('__'):
                A = getattr(obj,k)
                if Qtype(A)=='numpy.ndarray':
                    M += A.nbytes
                    print(text+'.'+k+' has',int(0.5+A.nbytes/(2**20)),'MB')
                elif Qtype(A).startswith('utils.') or Qtype(A)=='dict':
                    print(text+'.'+k)
                    M += Qmem(A,text+'  ')
    print(text+'TOTAL MEMORY:',int(0.5+M/(2**20)), 'MB')
    return M

# old and lame:
##def Qmem(*objs):
##    mems = []
##    for obj in objs:
##        if Qtype(obj) == 'numpy.ndarray':
##            mem = obj.nbytes
##        else:
##            mem = 0
##            for na in Qgetallnumpy(obj):
##                a = getattr(obj,na)
##                mem += a.nbytes
##                print(Qstr(obj),'takes',a.nbytes,'bytes')
##        print('TOTAL:',Qstr(obj),'takes',mem,'bytes')
##        mems.append(mem)
##    return mems

def Qmelt(*objs):
    for obj in objs:
        melt = {}
        for na in Qgetallnumpy(obj):
            a = getattr(obj,na)
            melt[na] = (a.tolist(),a.dtype,a.shape) # a.shape controversial as it won't change
        obj.__Q__melt = melt
            
def Qunmelt(*objs):
    for obj in objs:
        for na,qm in obj.__Q__melt.items():
            setattr(obj, na, np_array(qm[0],dtype=qm[1]))
        del obj.__Q__melt
    
    

def Qpy(py):
    f = os_getcwd()+'/'+py+'.py'
    subprocess_run('python '+f)


class QC:
    def __init__(self, o, s):
        d = {I[-1]:Q(o, ' '.join(I)) for I in Qexpr(s)}
        for k,v in d.items(): setattr(self, k, v)
    def __call__(self):
        return self.__dict__

Q_EXPR_REGEX = re.compile("[a-zA-Z_][a-zA-Z_0-9]*|[<>,:!\?\|]")
def Qexpr(s):
    def add(O,stack,name,get,call):
        if not stack: Qer('Qexpr hit empty stack.')
        comm = get+call
        if len(comm)>1: Qer('Qexpr too many commands:',comm)
        k = name if name else stack[-1]
        S = ' '.join(stack)
        if comm: S += comm
        if k in O: Qer('Qexpr collision, key:',k)
        O[k] = S
    def err(s):
        Qer('Misplaced',s,'in Qexpr.')
    Z = '<>,!?'
    L = re.findall(Q_EXPR_REGEX, s)
    # validate <>, here
    S = []
    N = ''
    G = ''
    C = ''
    B = ''
    O = {}
    for i in L+[',']:
        if i == ',':
            add(O,S,N,G,C)
            S = []
        elif i == '<':
            if not B: add(O,S,N,G,C)
            B = '<'
            S.pop()
        elif i == '>':
            comm = G+C
            if comm: Qer('Qexpr cannot > with comm:',comm)
            add(O,S,N,G,C)
        elif i == ':': N = ':'
        elif i == '!': C = '!'
        elif i == '?': G = '?'
        else:
            if N == ':': N = i
            elif N: Qer('Qexr naming confusion at:',N)
            else: S.append(i)
        if i in ',<>': C = ''; G = ''; N = ''
        if i != '<': B = ''
    return O
    
def Qexpr____OLD(s):
    def err(): Qer('Bad Qexpr s!')
    if not type(s) is str: err()
    z = []
    a = ''
    for c in s:
        if c.isspace():
            if a:
                z.append(a)
                a = ''
        elif c in '<>':
            #if c == '>' and z and z[-1]=='>': err()
            if a:
                z.append(a)
                a = ''
            z.append(c)
        else: a += c
    if a: z.append(a)
    if not z: return []
    if z[-1] in '<>': err()
    X = [[]]
    last = ''
    for y in z:
        if y =='>':
            if last in '<>': err()
            X.append(X[-1].copy())
        elif y=='<':
            if last == '>': err()
            if last != '<': X.append(X[-1].copy())
            if X[-1]: X[-1].pop()
            else: err()
        else:
            X[-1].append(y)
        last = y
    DICT = {x[-1]:x for x in X} # TO_DO: does not resolve duplicate aliases.
    return X
if 0:
    Qexpr(1,'person arm hand>finger1<thumb<<elbow<<<room chair>leg<seat<backrest<<lamp>bulb<stand')

def Qvomit(tree, dest):
    if Qtype(tree) in 'list set tuple'.split():
        for i in tree: Qvomit(i, dest)
    elif Qisdict(tree):
        for k in tree: Qvomit(tree[k], dest)
    else: dest.append(tree)

def Qha(obj, s):
    return hasattr(obj, '__'+s+'__') or hasattr(obj, s)
def Qga(obj, s):
    ga = getattr(obj, '__'+s+'__')
    if ga: return ga
    return getattr(obj, s)
def Qqa(obj, s): return Qga(obj, s) if Qha(obj, s) else ''
def Qstrappend(obj, blank=' '):
    s = str(obj)
    if s: return blank + s
    else: return ''

qnumtypes = sum([[a+b for a in 'int uint float'.split()] for b in '8 16 32 64'.split()],[])
def Qtype(obj): return str(type(obj))[8:-2]
def Qisdict(obj): return Qtype(obj) in ['dict','collections.OrderedDict']
def Qfunc(func): # magic
    c = func.__code__
    return ['@',c.co_name,'('+', '.join(c.co_varnames[:c.co_argcount])+')']
def Qstr(obj, fast=False):
    if hasattr(obj,'__Qstr__'): return obj.__Qstr__()
    t = Qtype(obj)
    if t in ['int','float']: return '#'+str(obj)
    if t == 'str':
        l = len(obj)
        max_str = 30
        if l > max_str:
            e = '$...'+str(l)
            return '`'+obj[0:(max_str-len(e))]
        return '`'+obj
    if t in ['NoneType','bool']: return '^'+str(obj)
    if t == 'type': return '^'+str(obj)[8:-2]
    if t in ['tuple','list','set'] or Qisdict(obj):
        t = '*'+t.capitalize()+'('+str(len(obj))+')'
        if len(obj): t += ' ['+Qstr(tuple(obj)[0])+'...'
        return t
    if t.startswith('numpy.'):
        t = t[6:]
        if t == 'ndarray':
            out = '*'+str(obj.dtype).capitalize()+'('+','.join(str(n) for n in obj.shape)+')'
            if obj.size == 0: out += '[EMPTY]'
            elif not fast:
                F = '%.2f' if 'float' in str(obj.dtype) else '%d'
                try: m = min(obj.flat); M = max(obj.flat)
                except: m = None; M = None
                if m == None: out += '[range==??]'
                elif m==M: out += '[=='+F%m+']'
                else: out += '['+F%m+chr(175)+F%M+']'
            return out
        if t in qnumtypes:
            return '#'+str(obj)+'('+t+')'
    if t in ['function','method']: return ''.join(Qfunc(obj))
    t = '?'+t
    if Qha(obj,'len'): t += '(%d)'%len(obj)
    return t
def Qissimple(obj):
    T = Qtype(obj)
    if T in 'numpy.ndarray tuple list set int float str NoneType bool type function method'.split(' ') \
       or T.startswith('numpy.') and T[6:] in qnumtypes: return True
    return False
        
            
    

##def Qhashable(obj):
##    T = {str:'$',int:'#',float:'#'}
##    for L in [True, False, None]: T[type(L)] = '^'
##    return T.get(type(obj),'')
##
##def Qval(obj):
##    return Qhashable(obj)+str(obj)


def Qtree(obj, search = ''):
    for L in Qbranch(obj,  '%', search):
        print(L)

def Qbranch(obj, name, search = ''):
    SH = []
    if Qisdict(obj):
        return [Qbranch(v, name+':'+Qstr(k)+' -> ', search) for k,v in obj.items()]
##    if Qtype(Qqa(obj,'init'))=='method':
##        out = []
##        for d in dir(obj):
##            a = getattr(obj,d)
##            o =  None
##            if Qtype(a) == 'method':
##                o = ''.join(Qfunc(a))
##            elif not(d.startswith('__') and d.endswith('__')):
##                o  = Qstr(d)+'.'+Qstr(a)
##            if type(o) is str: out.append(name+'.'+o)
##        return out
    return [name+Qstr(obj)]



def Qin(obj):
    if Qisdict(obj): return list(obj.keys())
    out = []
    for d in dir(obj):
        if d.startswith('__') and d.endswith('__'):
            if Qtype(getattr(obj,d)) == 'method': out.append(d)
        else: out.append(d)
    return out

# stay with M protect A from alliance: zadnych spolek

def Qgetall(obj, q):
    if type(q) is str: q = Qexpr(q)
    out = {}
    for k,v in q.items():
        r = Qroot(obj, v)
        if r: out[k] = r[1]
    return out
        
def Qsetall(obj, _dict):
    if Qisdict(obj): obj.update(_dict)
    else:
        for k,v in _dict.items(): setattr(obj, k, v)
    
def Qget(obj, s, root = '%'):
    comm = ''
    for e in '!?':
        if s.endswith(e):
            comm += e
            s = s[:-1]
    if len(comm)>1: Qer('Qroot command supplement too long:',comm)
    D = Qisdict(obj)
    if comm == '?':
        if D: return (s,obj.get(s))
        else: Qer('Qget cannot .get() from',Qstr(obj))
    i = Qin(obj)
    lo = s.lower()
    DIR = obj if D else dir(obj)
    if s in DIR: hits = [s]
    else:
        hits = [k for k in DIR if lo in k.lower()]
        if len(hits)>1: hits = [k for k in DIR if s in k]
    if len(hits) == 1:
        h = hits[0]
        ans = obj[h] if D else getattr(obj,h)
        if comm == '!': ans = ans()
        return (h,ans)
    elif len(hits) == 0:
        return None
##        print(s, 'not found in', root,'- contents:')
##        for _ in i: print('\t',_)
##        Qer('Thing not found.')
    else:
        print(s, 'found multiple times in', root,'- contents:')
        for _ in i: print('\t',_)
        print('hits:',hits)
        Qer('Thing not found.')

def Qroot(obj, s=''):
    steps = s.strip().split()
    if not steps: return '%', obj
    O = obj
    root = '%'
    for i in steps:
        g = Qget(O, i, root)
        if not g: return None
        sep = ':' if Qisdict(O) else '.'
        root += sep+g[0]
        O = g[1]
    return (root, O)


##Q>GLOBAL
##Q>"MAP NWK>TRs"
##Q>"PHYSICAL road_width_m|rwm"
##(self, locals()) + Q #def __radd__(self,a):
##OR: self + Q, locals() + Q
##Q>"|" # reset?

# Problems with methods - how to pass self?


class Q_OPERATOR:
    def default(self, global_obj, expr):
        self.global_obj = global_obj
        if type(expr) == str: expr = Qexpr(expr)
        self.global_comm = expr
    def __matmul__(self, dest):
        Qsetall(dest, Qgetall(self.global_obj, self.global_comm))
        #else: Qer('Q@ call failed with type',Qstr(x))
    def __init__(self):
        self.global_comm = {}
        self.global_obj = None
        self.last_root = None
        self.last_comm = None
    def __call__(self, x=None, y=None):
        if x is None: return self.last_comm
        if type(x) is str: self.last_comm = self.Q(self.last_obj, x, True)
        else:
            self.last_obj = x
            if y is None: self.last_comm = x
            else: self.last_comm = self.Q(x,y,True)
        return self.last_comm
    def Q(self,obj, s='',silent=True):
        r = Qroot(obj, s)
        if not r:
            if not silent: print(s,'not found.')
            return None
        R, O = r
        I = Qin(O)
        if not silent: print(R,'  is  ',Qstr(O),end='')
        if Qissimple(O) and not silent: print(' .')
        elif not silent:
            if not I: print('   , [empty].')
            else:
                print('   , containing:')
                tab = max(len(i) for i in I) + 4
                for i in I:
                    print(' '*8+i+' '*(tab-len(i))+Qstr(Qget(O,i)[1]))
        self.last_comm = O
        return O
    def push(self, x, silent):
        if Qtype(x) == 'str':
            self.last_comm = self.Q(self.last_obj,x,silent)
            return self.last_comm
        else:
            self.last_obj = x
            if not silent: print(Qstr(x))
    def __lt__(self, x): self.push(x,False)
    def __gt__(self, x): self.push(x,True)
    def __lshift__(self, x): return self.push(x,False)
    def __rshift__(self, x): return self.push(x,True)
    
    #def __invert__(self, x): return self()
    
Q = Q_OPERATOR()


if 1:
    class Koala:
        def __init__(self):
            self.be = 0
            self.d = {'a':5,'b':6}
        def x(self):
            return self.be + 2
    K = Koala()
##    D = {'K':K,'L':[[6,7],[9,8]]}
##    x = Qaccess(D, 'K d b')
##    print(x)

##    L = [4,5,6,7,8]
##    print(Qtype(L))
##    print(Qstr(L))
####    print(Qstr(tuple(L)))
##    from numpy import array, int8
####    A = array([[4,5],[5,6]])
####    print(Qstr(A))
####    I = int8(8)
####    print(Qstr(I))
####    print(Qstr(int(I)))
##    D = {5:'sdf','rg':{int:None,None:True}}
####    print(Qstr(D))
##
##    K = Koala()
##    print(Qtype(K))
##    print(Qfunc(Qbranch))
##    print('#'*50)
##    Qtree(K)
##    Qtree(D)
    


############################################### Q_LANG #############################

def QQ(code_line,LOCALS):
    P = code_line.split('..')
    if len(P) != 2:
        print('PFFT!')
        return None
    o, a = P 
    if o not in LOCALS:
        print('PFFFFFFFFFT!')
        return None
    if a in dir(LOCALS[o]):
        out = o+'.'+a
    else:
        out = o+"['"+a+"']"
    print(out)
    return out



##import ast
##
##def Qasttype(obj):
##    t = Qtype(obj)
##    if t.startswith('_ast.'): return t[5:]
##    else: return False
##
##def Qast(obj):
##    at = Qasttype(obj)
##    if not at: return False
##    print(at)
##    for k,v in obj.__dict__.items():
##        if Qasttype(v):
##            print(k,':')
##            Qast(v)
##        elif type(v) is list:
##            print(k,'...')
##            for i in v: Qast(i)
##        elif k in 'n s id attr'.split(): print(k,'=',v)
##
##def Qparse(code_line):
##    A = ast.parse(code_line).body[0]
##    Qast(A)
##    return A.body
    
    

#A = Qparse("x,y,z = 1,'2',3")




class Synonyms:
    def __init__(self, table):
        self.syn = {}
        for line in table.strip().split('\n'):
            ls = line.strip()
            if ls:
                words = ls.split()
                self.syn[words[0]] = True
                if len(words) > 1: 
                    for word in words[1:]: self.syn[word] = words[0]
    def __call__(self, word):
        s = self.syn.get(word)
        if s is None: return False
        if s is True: return word
        return self.syn[word]
        


##TASKS = Synonyms("""
##function def call method
##file
##folder
##local variable
##class
##""")


def Qndarray(t,x,fill=False):
    l = hasattr(x, '__len__')
    if fill is not False:
        if l: sh = tuple(x)
        else: sh = (x,)
        if fill is None: return np_empty(sh,t)
        if fill == 0: return np_zeros(sh,t)
        return np_ones(x,t)*fill
    if Qtype(x) == 'numpy.ndarray': return x.astype(t)
    if l: return np_array(x,t)
    if Qtype(x) == 'generator': return np_array(tuple(x),t)
    return np_zeros(x,t)

def Qf64(x, fill = False): return Qndarray('float64',x,fill)
def Qf32(x, fill = False): return Qndarray('float32',x,fill)
def Qi64(x, fill = False): return Qndarray('int64',x,fill)
def Qi32(x, fill = False): return Qndarray('int32',x,fill)
def Qi16(x, fill = False): return Qndarray('int16',x,fill)
def Qi8(x, fill = False): return Qndarray('int8',x,fill)
def Qu64(x, fill = False): return Qndarray('uint64',x,fill)
def Qu32(x, fill = False): return Qndarray('uint32',x,fill)
def Qu16(x, fill = False): return Qndarray('uint16',x,fill)
def Qu8(x, fill = False): return Qndarray('uint8',x,fill)
def Qbool(x, fill = False): return Qndarray('bool',x,fill)

def Qmx(s): return s.strip().split()

def Ql(a): return a.reshape([a.size])

def Qr(*rng):
    if hasattr(rng[0], '__len__'):
        if len(rng)==1: return range(len(rng[0]))
        Qer('Qr problem!')
    return range(*rng)

def Qp(*to_print): print(*to_print)
def Qer(*to_print): raise Exception(*to_print)


def Qdir(obj):
    if Qisdict(obj): return list(obj.keys())
    else: return {s for s in dir(obj) if not (s.startswith('__') and s.endswith('__'))}

def Qset(obj, s, val):
    if Qisdict(obj): obj[s] = val
    else: setattr(obj,s,val)

Q_SAVE_APPLY = {'Lf32':LoLf32,'Lu8':LoLu8,'Lu32':LoLu32,
                'f32':Qf32,'u32':Qu32}
def Qsave(dest, src, command='*'):
    COMM = Qmx(command)
    if COMM[0]=='*':
        S = set(Qdir(src))
        for e in COMM[1:]:
            if e in S: S.remove(e)
            else: Qer('Qsave: bad item',e,'in exceptions.\n',S)
        for n in S:
            Qset(dest, n, Qget(src, n)[1]) # ! tuple2
    # implement also name changing: "old_x:x@Lu8"
    else:        
        for C in COMM:
            P = C.split('@')
            v = Qget(src, P[0])[1] # ! tuple2
            if len(P) == 1: Qset(dest,P[0],v)
            elif len(P) == 2: Qset(dest,P[0],Q_SAVE_APPLY[P[1]](v))
            else: Qer("Bad Qsave item:",C)


def QD(obj, start = ''):
    d = Qdir(obj)
    t = Qisdict(obj)
    print(Qstr(obj), 'has', len(d), 'items:')
    out = []
    for i in d:
        s = Qstr(obj[i] if t else getattr(obj,i))
        if s.startswith(start): out.append((i,s))
    if not out: print('[NOTHING]')
    else:
        L = max(len(i) for i,s in out)
        for i,s in out: print(i, ' '*((L-len(i))+4), s)

    

def Qisname(name):
    return name.isidentifier() and not keyword_iskeyword(name)

# ! does not catch formatting errors:
Q_NEW_FORMAT = {'^T':True, '^F':False, '^N':None,
                '*L':LoL,
                '<>':set,'[]':list,'{}':dict,'()':tuple}
Q_NEW_FORMAT.update({'#%d'%n:n for n in range(10)})
def Qnew(obj, expr):
    T = '%' # uninitialized
    for v in Qmx(expr):
        n = v
        if len(v)>2 and v[-2:] in Q_NEW_FORMAT:
            T = Q_NEW_FORMAT[v[-2:]]
            n = v[:-2]
        if T=='%': Qer('Qnew needs initial type.')
        if not Qisname(n): Qer('Qnew: bad expression:',v)
        Qset(obj, n, T() if Qtype(T)=='type' else T)

# OLDER VERSION:
##Q_INIT_VALUES = {'[':lambda:list(),'{':lambda:dict(),'<':lambda:set(),
##                 '+':True,'-':False,'?':None,'0':0,'1':1}
##def Qinit(dest, command):
##    for C in Qmx(command):
##        n = C[:-1]; v = C[-1]
##        if n.isidentifier() and v in Q_INIT_VALUES:
##            v = Q_INIT_VALUES[v]
##            if Qtype(v) == 'function': v = v()
##            Qset(dest,n,v)
##        else: Qer('Bad item',C,'in Qinit().')




