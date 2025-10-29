### New routines in development to be integrated into the code hierarchy "soon"

##
##from odf.opendocument import OpenDocumentText
##from odf.style import Style, TextProperties
##from odf.text import H, P, Span
##
##textdoc = OpenDocumentText()
### Styles
##s = textdoc.styles
##h1style = Style(name="Heading 1", family="paragraph")
##h1style.addElement(TextProperties(attributes={'fontsize':"24pt",'fontweight':"bold" }))
##s.addElement(h1style)
### An automatic style
##boldstyle = Style(name="Bold", family="text")
##boldprop = TextProperties(fontweight="bold")
##boldstyle.addElement(boldprop)
##textdoc.automaticstyles.addElement(boldstyle)
### Text
##h=H(outlinelevel=1, stylename=h1style, text="My first text")
##textdoc.text.addElement(h)
##p = P(text="Hello world. ")
##boldpart = Span(stylename=boldstyle, text="This part is bold. ")
##p.addElement(boldpart)
##p.addText("This is after bold.")
##textdoc.text.addElement(p)
##textdoc.save("myfirstdocument.odt")
##





















# utils:
def summary(obj):
    s = str(type(obj))[8:-2]
    if hasattr(obj,'__len__'): s += '#'+str(len(obj))
    else:
        if hasattr(obj,'shape'): s += '#'+str(obj.shape)
    return s
def short(obj):
    print('<'+summary(obj)+'>')

def f(a,b,c):
    return a+b+c

def g(a,b,c=5):
    return a*b*c, 2


# https://medium.com/hackernoon/adding-a-pipe-operator-to-python-19a3aa295642
# https://pypi.org/project/pipe/

def apply(call, params):
    V = call.__code__.co_varnames # magic!
    P = {p:params[p] for p in V}
    print('@'+call.__name__,':',', '.join(v+'='+summary(P[v]) for v in V))
    return call(**P)
def pipe(calls, params, overwrite = False):
    for call in calls:# or enumerate and return index of function?
        if hasattr(call, '__len__'):
            func = call[0]
            if len(call) > 1: ret = call[1:]
            else: ret = ['@' + func.__name__]
        else:
            func = call
            ret = ['@' + func.__name__]
        if f in params and not overwrite:
            print(f,'dulpicate CALL!')
            print('PIPE ended prematurely')
            return f
        params[f] = apply(call, params)
    return None

D = {'a':4,'b':9,'c':2}
pipe([f,g,(f,'x'),(g,'y','z')],D)
print(D)

#TODO
# calls:
# len=1: function => @name
# len=2: function => str
# len>2: function => unpack tuple
# at the end, return calls that failed and resources that were missing.
# maybe an extra function that can apply functions out of order, or indeed validate call/param sequence
# also, need: call.__defaults__ - tuple from end

