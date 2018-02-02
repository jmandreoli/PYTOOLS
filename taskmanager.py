# File:                 taskmanager.py
# Creation date:        2013-11-18
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Simple task utilities
#

import logging, os, re, icalendar, uuid, sys
from datetime import datetime, timedelta
from copy import deepcopy
from pytz import timezone, utc
from contextlib import contextmanager
from email.utils import parseaddr, getaddresses
from email.message import Message
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parseaddr, getaddresses
from lxml.etree import _ElementTree as xmlElementTree, ElementTextIterator as xmlElementTextIterator
from lxml.builder import ElementMaker as xmlElementMaker
from lxml.html import tostring as tohtml, fromstring as fromhtml
from smtplib import SMTP
from base64 import b64encode

logger = logging.getLogger(__name__)

#==================================================================================================
def sendmail(message,login=None,pwd=None,smtpargs=None,**ka):
  r"""
:param message: email message
:type message: :class:`email.message.Message`
:param login: user login on SMTP server
:type login: :class:`str`
:param pwd: password on SMTP server
:type pwd: :class:`str`
:param smtpargs: arguments passed to the :class:`smtplib.SMTP` to access the SMTP server (host, port...)
:type smtpargs: :class:`Dict[str,any]`
  """
#==================================================================================================
  def recipients(msg): return set(email for name,email in getaddresses([v for h in ('to','cc','bcc') for v in msg.get_all(h,())]))
  def sender(msg): return parseaddr(msg.get('from'))[1]
  if not message.get('date'): message['date'] = datetime.now().astimezone().strftime('%a, %d %b %Y %H:%M:%S %z')
  ka.update(from_addr=sender(message),to_addrs=recipients(message),msg=message.as_string())
  with SMTP(**smtpargs) as s:
    s.starttls()
    s.login(login,pwd)
    return s.sendmail(**ka)

#==================================================================================================
def announcement(shorttxt=None,plaintxt=None,html=None,icscal=None,filename=None,charset='utf-8'):
  r"""
:param icscal: calendar version of the announcement
:type icscal: :class:`icalendar.Calendar`
:param plaintxt: plain text version of the announcement
:type plaintxt: :class:`str`
:param html: html version of the announcement
:type html: :class:`lxml.etree._ElementTree`
:param shorttxt: short (1 liner) text version of the announcement
:type shorttxt: :class:`str`
:param filename: file name associated to the ics calendar in the announcement
:type filename: :class:`str`
:param charset: charset encoding specification
:type charset: :class:`str`
:rtype: :class:`email.message.Message`

Returns the announcement specified in different forms (*shorttxt*, *plaintxt*, *xhtml*, *icscal*) as an email message.
  """
#==================================================================================================
  checktype('shorttxt',shorttxt,str)
  checktype('plaintxt',plaintxt,str)
  checktype('html',html,xmlElementTree)
  checktype('icscal',icscal,icalendar.Calendar)
  msg = MIMEMultipart('alternative')
  msg.set_param('name','announce.eml')
  msg.add_header('subject',shorttxt)
  msg.attach(MIMEText(plaintxt,'plain',charset))
  msg.attach(MIMEText(tohtml(html,doctype='<!DOCTYPE html>'),'html',charset))
  m = MIMEText(icscal.to_ical().decode('utf-8'),'calendar',charset)
  m.set_param('method',icscal.get('method'))
  if filename is not None: m.add_header('content-disposition','attachment',filename=filename+'.ics')
  msg.attach(m)
  return msg

#==================================================================================================
def calendar(content=None,method='PUBLISH',version='2.0',reminder=None):
  r"""
:param content: list of events
:type content: :class:`List[icalendar.Event]`
:param reminder: reminder
:type reminder: :class:`datetime.timedelta`
:param method: ics method for the icalendar
:type method: :class:`str`
:param version: icalendar version;should not be changed
:type version: :class:`str`
:rtype: :class:`icalendar.Calendar`

Return a calendar object listing all the events in *content*, possibly augmented, for those confirmed, with a reminder specified by *reminder* (delta from start, negative).
  """
#==================================================================================================
  for evt in content: checktype('content elements',evt,icalendar.Event)
  checktype('method',method,str)
  checktype('version',version,str)
  checktype('reminder',reminder,timedelta,allow_none=True)
  cal = icalendar.Calendar()
  cal.add('prodid','-//ChronoManager//0.1')
  cal.add('version',version)
  cal.add('method',method)
  alarm = None
  if reminder is not None:
    rmdr = icalendar.vDuration(reminder)
    rmdr.params['related'] = icalendar.vText('START')
    alarm = icalendar.Alarm()
    alarm.add('action','DISPLAY')
    alarm.add('description','REMINDER')
    alarm.add('trigger',rmdr,encode=False)
  for evt in content:
    if evt.get('status')=='CONFIRMED' and alarm is not None:
      evt.add_component(alarm)
    cal.add_component(evt)
  return cal

#==================================================================================================
def event(start=None,duration=None,permalink=None,sequence=0,confirmed=True,priority=5,klass='PUBLIC',transp='OPAQUE'):
  r"""
:param start: event start instant, in UTC
:type start: :class:`datetime.datetime`
:param duration: event duration
:type duration: :class:`datetime.timedelta`
:param confirmed: whether the event is confirmed
:type confirmed: :class:`bool`
:param permalink: permanent url for the event
:type permalink: :class:`str`
:rtype: :class:`icalendar.Event`

Returns a calendar event, which can be further extended. The *permalink*, if not const:`None`, is used to generate a unique ID of the event.
  """
#==================================================================================================
  checktype('start',start,datetime)
  checktype('duration',duration,timedelta)
  checktype('permalink',permalink,str,allow_none=True)
  checktype('sequence',sequence,int)
  checktype('confirmed',confirmed,bool)
  checktype('priority',priority,int)
  checktype('klass',klass,str)
  checktype('transp',transp,str)
  start = start.astimezone(utc)
  evt = icalendar.Event()
  evt.add('dtstamp',utc.localize(datetime.utcnow()))
  evt.add('dtstart',start)
  evt.add('duration',duration)
  evt.add('status','CONFIRMED' if confirmed else 'TENTATIVE')
  evt.add('sequence',sequence)
  evt.add('priority',priority)
  evt.add('class',klass)
  evt.add('transp',transp)
  if permalink is not None:
    uid = uuid.uuid3(uuid.NAMESPACE_URL,permalink)
    evt.add('uid',uid)
  return evt

#--------------------------------------------------------------------------------------------------
def xmlpuretext(e,pat=re.compile('(\n{2,})',re.UNICODE)):
  r"""
:param e: XML node
:type e: :class:`lxml.etree.ElementBase`

Returns the text content of *e*, contracting multiple consecutive newlines into a single one.
  """
#--------------------------------------------------------------------------------------------------
  return pat.sub('\n',''.join(xmlElementTextIterator(e)))

#--------------------------------------------------------------------------------------------------
def xmlsubstitute(doc,d):
  r"""
:param doc: target XML document
:type doc: :class:`lxml.etree.ElementBase`
:param d: substitution table
:type d: :class:`Dict[str,lxml.etree.ElementBase]`

Replaces each node of the form ``<?parm xx?>`` in *doc* by the value of *d* at key ``xx``, if it exists.
  """
#--------------------------------------------------------------------------------------------------
  for parm in doc.xpath('//processing-instruction("parm")'):
    x = d.get(parm.text)
    if x is None:
      parm.getparent().remove(parm)
    else:
      x = deepcopy(x)
      x.tail = parm.tail
      parm.getparent().replace(parm,x)
  return doc

#--------------------------------------------------------------------------------------------------
def safedump(content,path,mode='w'):
  r"""
:param content: target string
:type content: :class:`Union[str,bytes]`
:param path: target path
:type path: :class:`str`

Saves string *content* into file at *path* and creates a copy of the old file.
  """
#--------------------------------------------------------------------------------------------------
  if os.path.exists(path):
    dirn,fn = os.path.split(path)
    pathnew,pathsav = os.path.join(dirn,'new-'+fn),os.path.join(dirn,'bak-'+fn)
    with open(pathnew,mode) as v: v.write(content)
    if os.path.exists(pathsav): os.remove(pathsav)
    os.rename(path,pathsav)
    os.rename(pathnew,path)
  else:
    with open(path,mode) as v: v.write(content)

#--------------------------------------------------------------------------------------------------
def odformfill(doc,**param):
  r"""
:param doc: xml document representing the content of a word document containing form fields
:type doc: :class:`lxml.etree._ElementTree`
:param param: an assignment of the form fields to values
:type param: :class:`Dict[str,Union[str,List[str]]]`

Updates *doc* so that it represents the same word document with the form fields present as keys in *param* assigned their values. If *param* has keys which are not form field labels, an error is raised. This function works only with simple fields having text or multiline text values.
  """
#--------------------------------------------------------------------------------------------------
  root = doc.getroot()
  wns = root.nsmap['w']
  E = xmlElementMaker(namespace=wns,nsmap=dict(w=wns))
  def xpath(e,path,single=False,nsmap=dict(w=wns)):
    r = e.xpath(path,namespaces=nsmap)
    if single:
      if len(r) != 1:
        raise Exception('More than one instance found of: {}'.format(repr(path)))
      r = r[0]
    return r
  for sdt in xpath(root,'descendant::w:sdt'):
    tag = xpath(sdt,'w:sdtPr/w:tag/@w:val',single=True)
    content = xpath(sdt,'w:sdtContent',single=True)
    val = param.pop(tag,None)
    if val is not None:
      if isinstance(val,(list,tuple)):
        val = tuple(e for x in val for e in (E.t(x),E.br()))[:-1]
      else:
        val = E.t(val),
      p = xpath(content,'w:tc/w:p')
      if len(p)==1: content = p[0]
      elif len(p)>1: raise Exception('Form field type not recognised: {}'.format(tag))
      del content[:]
      content.append(E.r(*val))
  if param: raise Exception('Keys not found in form: {}'.format(','.join(param.keys())))

#--------------------------------------------------------------------------------------------------
def mail2str(message):
  r"""
:param message: email message
:type message: :class:`email.message.Message`

Returns the content of *message* as a pretty string. All headers are displayed. Only the content of attachments of mimetype ``text/plain``.
  """
#--------------------------------------------------------------------------------------------------
  def disp(msg,idn='',iz=''):
    if idn: yield '{} Part[{}]'.format(iz,idn); iz += '  '
    for k,v in msg.items(): yield '{} {}: {}'.format(iz,k.capitalize(),v)
    if msg.is_multipart():
      for n,msg1 in enumerate(msg.get_payload(),1):
        yield from disp(msg1,'{}{}.'.format(idn,n),iz)
    else:
      content = msg.get_payload(decode=True)
      q = '{} Content (size: {}):'.format(iz,len(content))
      mime = msg.get_content_type()
      if mime=='text/plain':
        yield q
        d = 76-len(iz)
        for line in content.decode(str(msg.get_charset())).split('\n'):
          yield '{} >>> {}'.format(iz,line[:d])
          for k in range(d,len(line),d): yield '{} ... {}'.format(iz,line[k:k+d])
      else: yield q+' hidden'
  checktype('message',message,Message)
  return '\n'.join(disp(message))

#--------------------------------------------------------------------------------------------------
def mail2html(message):
  r"""
:param message: email message
:type message: :class:`email.message.Message`

Returns the content of *message* as a pretty html object (as understood by lxml). All headers are displayed. Only the content of attachments of mimetypes usually accepted by browsers are displayed.
  """
#--------------------------------------------------------------------------------------------------
  from lxml.html.builder import E
  def disp(msg):
    if msg.is_multipart():
      payload = E.table(E.tbody(*(E.tr(E.td('[{}]'.format(n)),E.td(disp(msg1)),style='border-bottom: thin solid black') for n,msg1 in enumerate(msg.get_payload(),1))),style='border-collapse: collapse;')
      payload = E.tr(E.td(payload,colspan='2'))
    else:
      content = msg.get_payload(decode=True)
      mime = msg.get_content_type()
      t = '{} (size={})'.format(mime,len(content))
      if mime == 'text/plain': content = E.div(content.decode(str(msg.get_charset())),style='white-space: pre-line;')
      elif mime == 'text/html': content = E.div(*fromhtml(content.decode(str(msg.get_charset()))).xpath('//body/*'))
      elif mime.startswith('image/'): content = E.div(E.img(src='data:{};base64,{}'.format(mime,b64encode(content)),style='max-width:5cm; max-height:5cm;'))
      else: content = E.span('Hidden: {}'.format(t),style='background-color: gray; color: white;'); t = ''
      payload = E.tr(E.td(E.span('content',title=t)),E.td(content))
    return E.table(
      E.thead(*(E.tr(E.td(k,style='white-space: nowrap; font-weight: bold; font-size: xx-small;'),E.td(v)) for k,v in msg.items()),style='border: thick solid gray;'),
      E.tbody(E.tr(E.td(payload,colspan='2'))),
      border='1',style='border-collapse: collapse;'
    )
  checktype('message',message,Message)
  return disp(message)

#--------------------------------------------------------------------------------------------------
def htmlhidden(**ka):
#--------------------------------------------------------------------------------------------------
  from lxml.html.builder import E
  for k,v in ka.items(): yield E.input(type='hidden',name=k,value=v)

#--------------------------------------------------------------------------------------------------
def checktype(name,x,*typs,allow_none=False):
#--------------------------------------------------------------------------------------------------
  if not ((allow_none and x is None) or isinstance(x,typs)):
    raise TypeError('{} must be of type {}, not {}'.format(name,'|'.join(map(str,typs)),type(x)))
