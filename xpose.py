import os, falcon
from datetime import datetime
from lxml.html import fromstring as fromhtml, tostring as tohtml
from lxml.etree import parse as parsexml, fromstring as fromxml, tostring as toxml
from email.utils import parseaddr, getaddresses

class TestResource:
  def on_get(self,req,resp):
    def f(**ka): return str(ka)
    docr = E.html(
      E.head(E.title('test'),E.meta(charset='UTF-8')),
      E.body(E.pre(f(**req.params)),E.form(E.div(E.textarea(name='xx')),E.input(type='submit',value='xxx'),action='/test',method='get'))
    )
    resp.content_type = 'text/html'
    resp.body = tohtml(docr,pretty_print=True,encoding='utf-8',doctype='<!DOCTYPE html>')

class StaticResource:
  mimetypes = {'.txt':'text/plain','.html':'text/html','.js':'text/js','.css':'text/css','.png':'image/png','.jpg':'image/jpeg'}
  def on_get(self,req,resp,name):
    ext = os.path.splitext(name)[1]
    path = os.path.join('static',name)
    resp.content_type = self.mimetype.get(ext,'application/octet-stream')
    resp.stream, resp.stream_len = io.open(path,'rb'),os.path.getsize(path)

class ManagedResource:

  def on_get(self,req,resp,idn=None,op=None):
    from lxml.html.builder import E
    t = self.lookup(idn)
    proc = getattr(self,'view_'+op)
    docr = E.html(
      E.head(E.title(op),E.meta(charset='UTF-8')),
      E.body(proc(t,**req.params))
    )
    resp.content_type = 'text/html'
    resp.body = tohtml(docr,pretty_print=True,encoding='utf-8',doctype='<!DOCTYPE html>')

  def on_post(self,req,resp,idn=None,op=None):
    from lxml.html.builder import E
    t = self.lookup(idn)
    proc = getattr(self,'do_'+op)
    docr = E.html(
      E.head(E.title(op),E.meta(charset='UTF-8')),
      E.body(proc(t,**req.params))
    )
    resp.content_type = 'text/html'
    resp.body = tohtml(docr,pretty_print=True,encoding='utf-8',doctype='<!DOCTYPE html>')

  def view_edit(self,t):
    from lxml.html.builder import E
    return E.form(
      E.div(E.textarea(toxml(t.toxml(),encoding=str),name='content',rows='30',cols='80')),
      E.input(type='submit',value='save'),
      action='edit',method='post'
    )
  def do_edit(self,t,content=None):
    from lxml.html.builder import E
    xml = fromxml(content.replace('\r\n','\n'))
    xml.tail = t.xml.tail
    t.fromxml(xml)
    t.save()
    return E.div('{} has been updated'.format(t))

class ManagerResource (ManagedResource):
  def __init__(self,*L):
    self.ident = ''
    self.sourcename = 'manager'
    self.initsources(L)
  def __iter__(self): yield self
  def __repr__(self): return 'manager'
  def actions(self): yield 'publish'
  def lookup(self,idn):
    assert idn=='', idn
    return self
  def initsources(self,L):
    assert all([isinstance(r,ManagedResource) for r in L])
    self.sources = (self,)+L
    for i,r in enumerate(L,1): r.sourcename = '{}{:02x}'.format(r.factory.__name__,i)
  def initapp(self,app):
    def redirect(req,resp,R={'/':'/manager//list','/favicon.ico':'/static/favicon.png'}):
      url = R.get(req.path)
      if url is None: raise Exception('No matching url')
      resp.status = falcon.HTTP_SEE_OTHER
      resp.location = url
    for r in self.sources: app.add_route('/{}/{{idn}}/{{op}}'.format(r.sourcename),r)
    app.add_route('/static/{name}',StaticResource())
    app.add_route('/test',TestResource())
    app.add_sink(redirect)
    app.req_options.auto_parse_form_urlencoded=True
  def view_list(self,t):
    from lxml.html.builder import E
    assert t is self
    ref = datetime.now().date()
    def button(sn,idn,a):
      return E.button(a,type='button',onclick='this.form.action=\'/{}/{}/{}\';this.form.submit()'.format(sn,idn,a))
    return E.form(
      E.table(
        E.tbody(
          *(
            E.tr(
              E.td(repr(t)),
              E.td(button(r.sourcename,t.ident,'edit'),*(button(r.sourcename,t.ident,a) for a in t.actions()))
            )
            for r in self.sources for t in r
          )
        )
      ),
      method='get',target='_blank'
    )

  def view_publish(self,t):
    from lxml.html.builder import E
    assert t is self
    return E.form(
      E.div(E.b('mode'),E.input(type='text',name='mode',length='10')),
      E.input(type='submit',value='next'),
      action='publish',method='post',target='_blank'
    )

  def do_publish(self,t,mode=None):
    from lxml.html.builder import E
    assert t is self
    a = 'as_'+mode
    L = [getattr(t,a)() for r in self.sources for t in r if hasattr(t,a)]
    with open(mode,'wb') as v: v.write(getattr(self,a)(L))
    return ''

class ManagedXmlResource(ManagedResource):

  path = None
  factory = None
  qual = None
  qual1 = None
  document_ = None

  @property
  def document(self):
    doc = self.document_
    if doc is None:
      with open(self.path) as u: self.document_ = doc = parsexml(u)
      self.ns = dict(my=doc.getroot().nsmap[None])
    return doc

  def __iter__(self): return self.select(self.qual)
  def lookup(self,idn):
    e, = self.document.xpath(self.qual1.format(idn),namespaces=self.ns)
    return self.factory(e)
  def select(self,qual=None):
    yield from (self.factory(e) for e in self.document.xpath(qual,namespaces=self.ns))

class XmlEntry:
  def toxml(self): return self.xml
  def fromxml(self,e):
    self.xml.getparent().replace(self.xml,e)
    self.xml = e
  def save(self):
    doc = self.xml.getroottree()
    with open(doc.docinfo.URL,'wb') as v: v.write(toxml(doc,xml_declaration=True,encoding='utf-8'))
