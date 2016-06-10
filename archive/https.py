# File:			pyHTTPServer.py
# Creation date:	2009-10-01
# Contributors:		Jean-Marc Andreoli
# Language:		Python
# Purpose:		A generic HTTP request handler
#
# *** Copyright (c) 2005 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***

import os, sys, traceback
from cgi import parse_qs, parse_multipart, parse_header, escape
from urllib import quote, unquote
from BaseHTTPServer import BaseHTTPRequestHandler

#------------------------------------------------------------------------------
# class HTTPRequestHandler
#------------------------------------------------------------------------------

class HTTPRequestHandler (BaseHTTPRequestHandler):

  def parseinput(self,t,parms):
    if t == 'application/x-www-form-urlencoded':
      n = self.headers.get('content-length')
      if n is not None:
        self.arguments.update(parse_qs(self.rfile.read(int(n))))
    elif t == 'multipart/form-data':
      self.arguments.update(parse_multipart(self.rfile,parms))
    else:
      return True

  def map_request(self,isget=False,ispost=False):
    self.isget = isget
    self.ispost = ispost
    try:
      self.arguments = args = {}
      t = self.headers.get('content-type')
      if t is not None:
        t,parms = parse_header(t)
        try:
          if self.parseinput(t,parms):
            self.send_error(415)
            return
        except:
          self.send_error(500)
          raise
      p = self.path.split('?',1)
      if len(p)>1:
        for a,l in parse_qs(p[1]).items():
          ll = args.get(a)
          if ll is None: args[a] = l
          else: ll.extend(l)
      p = p[0]
      if p == '' or p == '/':
        if not self.isget: raise Exception('Missing path')
        return self.server.home(self)
      p = unquote(p).split('/')
      x = self.server
      for a in p[1:]:
        exc = a[0] == '!'
        if exc: a = a[1:]
        a = a.split('~',1)
        if a[0]: x = getattr(x,a[0])
        if len(a)>1: x = x.get(a[1])
        if exc: x = x(self)
      return x
    except:
      traceback.print_exc()
      doc,e = self.xmlreplydoc()
      describe_exception(e,sys.exc_info())
      return lambda: self.send_xmlreply(doc,reason='Uncaught exception')
    else:
      return x

  def getarg(self,a,v=None):
    x = self.arguments.get(a)
    return x[0] if x else v

  def getargs(self,a):
    return self.arguments.get(a,[])

  def do_GET(self):
    self.map_request(isget=True)()

  def do_POST(self):
    self.map_request(ispost=True)()

  def send_reply(self,content,type='text/html',rc=200,reason=None,hdr={}):
    self.send_response(rc,reason)
    for (k,v) in hdr.items(): self.send_header(k,v)
    self.send_header('content-type',type)
    self.send_header('content-length',str(len(content)))
    self.end_headers()
    if content:
      self.wfile.write(content)
      self.wfile.flush()

  def forward_reply(self,r):
    self.send_response(r.status,r.reason)
    for k,v in r.getheaders(): self.send_header(k,v)
    self.end_headers()
    self.wfile.write(r.read())
    self.wfile.flush()

  def send_xmlreply(self,content,type='text/xml',**a):
    if content is None: content,e = self.xmlreplydoc()
    self.send_reply(content.toxml(),type,**a)

  def xmlreplydoc(self,style=None,**kw):
    doc = self.server.createXMLDocument(**kw)
    e = doc.documentElement
    if e.namespaceURI: e.setAttribute('xmlns',e.namespaceURI)
    if style: doc.insertBefore(doc.createProcessingInstruction('xml-stylesheet','type="text/xsl" href="%s"'%style),e)
    return doc, e

#------------------------------------------------------------------------------
# class Repository
#------------------------------------------------------------------------------

class Repository:

  Mimetypes = {
    '.xml':  'text/xml',
    '.xsl':  'text/xsl',
    '.css':  'text/css',
    '.html': 'text/html',
    '.js':   'text/javascript',
    '.csv':  'text/csv',
    '.txt':  'text/plain',
    '.gif':  'image/gif',
    '.png':  'image/png',
    '.jpg':  'image/jpg'
    }

  class Raw:

    def __init__(self,content,mimetype):
      self.content = content
      self.mimetype = mimetype

    def __call__(self,req):
      assert req.isget
      return lambda: req.send_reply(self.content,self.mimetype)

  class RawDir:

    def __init__(self,parent,path):
      self.parent = parent
      self.path = path

    def get(self,n):
      return self.parent.get(os.path.join(self.path,n))

  def __init__(self,path,mimetypes=None,data={}):
    self.path = path
    if mimetypes is None: mimetypes = {}
    mimetypes.update(self.Mimetypes)
    self.mimetypes = mimetypes
    self.data = data

  def get(self,n):
    r = self.data.get(n)
    if r is not None: return r
    p = os.path.join(self.path,n)
    if os.path.isdir(p): return self.RawDir(self,n)
    t = self.mimetypes.get(os.path.splitext(n)[1],'application/octet-stream')
    with open(p) as u: return self.Raw(u.read(),t)

#------------------------------------------------------------------------------
# miscellanous
#------------------------------------------------------------------------------

def describe_exception(e,info):
  doc = e.ownerDocument
  exc = info[1]
  tb = traceback.extract_tb(info[2])
  e.setAttribute('exception',str(exc.__class__))
  e.setAttribute('message',str(exc))
  for s in tb:
      at = doc.createElement('at')
      e.appendChild(at)
      at.setAttribute('file',s[0])
      at.setAttribute('line',str(s[1]))
      at.setAttribute('function',s[2])
      if s[3] is not None: at.appendChild(doc.createTextNode(s[3]))
