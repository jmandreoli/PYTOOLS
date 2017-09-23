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
from email.message import Message
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from lxml.etree import parse as xmlparse, tostring as xml2str, ElementBase as xmlElementBase, ElementTextIterator as xmlElementTextIterator
from lxml.html import tostring as html2str
logger = logging.getLogger(__name__)

#==================================================================================================
def mailsend(smtphost,*L,confirm=False,from_addr=None,to_addrs=None,cc_addrs=None,bcc_addrs=None,**ka):
  r"""
:param smtphost: a SMTP host
:type smtphost: :class:`smtplib.SMTP`
:param L: a list of email messages
:type L: :class:`Iterable[email.message.Message]`
:param confirm: whether to ask the user for confirmation of each email
:type confirm: :class:`bool`
:param from_addr: email address for 'from' field
:type from_addr: :class:`str`
:param to_addrs: list of email addresses for 'to' field
:type to_addrs: :class:`List[str]`
:param cc_addrs: list of email addresses for 'cc' field
:type cc_addrs: :class:`List[str]`
:param bcc_addrs: list of email addresses for 'bcc' field
:type bcc_addrs: :class:`List[str]`

Appends from,to,cc,bcc fields to each email message in *L*, then invokes method :meth:`sendmail` of *smtphost* on each message with keyword arguments *ka*. If *confirm* is :const:`True`, the user is given the opportunity to confirm or cancel each sending. Returns a list of same length as *L* holding the status of each send operation. Status is:

* ``cancelled`` if the email was cancelled out by the user (only when *confirm* is :const:`True`),
* ``skipped`` if a previous operation raised an exception (so, after an exception, the status is either ``cancelled`` or ``skipped``),
* ``success`` if the operation completed successfully,
* ``partial`` if the operation was partially successful (but some recipients were not reached),
* ``failure`` if the operation raised an exception (all non ``cancelled`` subsequent operations will be ``skipped``).
  """
#==================================================================================================
  def base(addr,pat=re.compile('.*<(.*)>'),pat2=re.compile(r'(?:\w|[._-])+@(?:\w|[_-])+(?:[.](?:\w|[_-])+)+')):
    m = pat.fullmatch(addr)
    if m is not None: addr = m.group(1)
    m = pat2.fullmatch(addr)
    if m is None: raise Exception('Invalid email address')
    return addr
  for msg in L: checktype('argument',msg,Message)
  checktype('from_addr',from_addr,str)
  sender = base(from_addr)
  recipients = []
  header_addrs = []
  for hdr,addrs in (('to',to_addrs),('cc',cc_addrs),('bcc',bcc_addrs)):
    if addrs is None: continue
    for addr in addrs: checktype('{}_addrs elements'.format(hdr),addr,str)
    header_addrs.append((hdr,', '.join(addrs)))
    recipients.extend(base(addr) for addr in addrs)
  for msg in L:
    msg.add_header('from',from_addr)
    for hdr,addrs in header_addrs: msg.add_header(hdr,addrs)
  La = [dict(from_addr=sender,to_addrs=recipients,msg=msg.as_string(),**ka) for msg in L]
  if confirm:
    L_ = []; empty = True
    try:
      for i,msg in enumerate(L,1):
        print('\x1BcCONFIRM MAIL {}/{}'.format(i,len(L)))
        maildisplay(msg,('plain',))
        while True:
          try: r = input('Ret: OK; ^D: skip> ')
          except EOFError: L_.append(False); break
          if r.strip(): continue
          else: L_.append(True); empty = False; break
    finally: print('\x1Bc',end='',flush=True)
    if empty: raise Exception('All mails were cancelled out by user')
  else: L_ = len(L)*(True,)
  ncanc = nskip = nsucc = 0
  R = []; exc = None; Ls = []
  with smtphost as smtp:
    for confirmed,sendargs in zip(L_,La):
      if confirmed:
        if exc is None:
          try: err = smtp.sendmail(**sendargs)
          except Exception as exc_: status = 'failure'; exc = exc_
          else:
            if err: status = 'partial'; R.append(err)
            else: status = 'success'; nsucc += 1
        else: status = 'skipped'; nskip += 1
      else: status='cancelled'; ncanc += 1
      Ls.append(status)
  npart = len(R)
  nfail = 0 if exc is None else 1
  logger.info('Mail procedure report [submitted: %s | cancelled: %s, success: %s, partial: %s, failure: %s, skipped: %s]',len(L),ncanc,nsucc,npart,nfail,nskip)
  if R: logger.warn('Cause(s) of partial success: %s',set(R))
  if exc is not None: logger.warn('Cause of failure: %s',exc)
  if nsucc==0: raise Exception('No mail was successfully sent')
  return Ls

#==================================================================================================
def announcement(shorttxt=None,plaintxt=None,html=None,icscal=None,filename=None,charset='utf-8'):
  r"""
:param icscal: calendar version of the announcement
:type icscal: :class:`icalendar.Calendar`
:param plaintxt: plain text version of the announcement
:type plaintxt: :class:`str`
:param html: html version of the announcement
:type html: :class:`lxml.etree.ElementBase`
:param shorttxt: short (1 liner) text version of the announcement
:type shorttxt: :class:`str`
:param filename: file name associated to the ics calendar in the announcement
:type filename: :class:`str`
:param charset: charset encoding specification
:type charset: :class:`str`
:rtype: :class:`email.message.Message`

Returns the announcement specified in different forms (*shorttxt*, *plaintxt*, *html*, *icscal*) as an email message.
  """
#==================================================================================================
  checktype('shorttxt',shorttxt,str)
  checktype('plaintxt',plaintxt,str)
  checktype('html',html,xmlElementBase)
  checktype('icscal',icscal,icalendar.Calendar)
  msg = MIMEMultipart('alternative')
  msg.set_param('name','announce.eml')
  msg.add_header('subject',shorttxt)
  msg.attach(MIMEText(plaintxt,'plain',charset))
  msg.attach(MIMEText(html2str(html),'html',charset))
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
    alarm.add('trigger',rmdr,0)
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

#==================================================================================================
@contextmanager
def transaction():
  r"""
Simply encloses some code in transaction status messages.
  """
#==================================================================================================
  logger.info('Transaction started [%s].',datetime.now())
  try: yield
  except KeyboardInterrupt: print(flush=True); logger.warn('Transaction aborted by user.')
  except: print(flush=True); logger.warn('Transaction aborted.'); raise
  else: logger.info('Transaction completed.')

#==================================================================================================
@contextmanager
def xmlfile_transaction(path=None,updating=False,namespaces={},target=None):
  r"""
:param path: path of the file
:type path: :class:`str`
:param updating: whether the transaction is meant to perform changes on the xml content
:type updating: :class:`bool`
:param namespaces: a dictionary of prefix-namespace associations, for use in *target*
:type namespace: :class:`Dict[str,str]`
:param target: an Xpath expression for selecting the xml nodes
:type target: :class:`str`

Opens a transaction to perform operations on nodes of an XML file at *path*. The list of nodes selected by *target* is returned on entering this context. On exit, if *updating* is :const:`True`, the document loaded on entering, and which should have been updated, is dumped again into *path* (a copy of the old file is preserved).
  """
#==================================================================================================
  with transaction():
    doc = xmlparse(path)
    namespaces = dict((k,(doc.getroot().nsmap[None] if v is None else v)) for k,v in namespaces.items())
    L = doc.xpath(target,namespaces=namespaces)
    if L: yield L
    else: raise Exception('no matching entry found')
    if updating: safedump(xml2str(doc,encoding=doc.docinfo.encoding,xml_declaration=True),path,'wb')

#==================================================================================================
# Utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
def maildisplay(message,textshow=None):
  r"""
:param message: email message
:type message: :class:`email.message`
:param textshow: a list of subtypes of mimetype ``text`` to display (default: all subtypes)
:type textshow: :class:`List[str]`

Displays the content of *message*. All headers are displayed. Only the content of attachments of mimetype ``text`` and subtype in *textshow* are displayed.
  """
#--------------------------------------------------------------------------------------------------
  def struct(msg,iz=''):
    for k,v in msg.items(): print(iz,'%s: %s'%(k.capitalize(),v))
    if msg.is_multipart():
      iz += '  '
      for msg1 in msg.get_payload():
        for m in struct(msg1,iz): yield m
    else:
      if msg.get_content_maintype()=='text' and (textshow is None or msg.get_content_subtype() in textshow):
        status = 'shown below'
        yield msg
      else: status = 'hidden'
      print(iz,'<<<------------ Content',status,'------------>>>')
  checktype('message',message,Message)
  L = tuple(struct(message))
  if L:
    for i,msg in enumerate(L,1):
      print('<<<------------ Content of part ',i,'------------>>>')
      content = msg.get_payload(decode=True)
      print(content.decode(str(msg.get_charset())))

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
    if x is not None:
      x = deepcopy(x)
      x.tail = parm.tail
      parm.getparent().replace(parm,x)
  return doc

#--------------------------------------------------------------------------------------------------
def safedump(content,path,mode='w'):
  r"""
:param content: target string
:type content: :class:`str`
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
def select(L):
  r"""
:param L: list from which to select
:type L: :class:`Iterable[object]`

Prompts user for selection in a list *L*.
  """
#--------------------------------------------------------------------------------------------------
  L = list(L)
  n = len(str(len(L)))
  for k,v in enumerate(L,1): print('[{}] {}'.format(str(k).rjust(n),v))
  while True:
    r = input('Ret: all; *: select (space separated)> ').strip()
    if r:
      try: return [L[int(k.strip(),10)-1] for k in r.split()]
      except KeyboardInterrupt: raise
      except: print('Invalid choice'); continue
    else: return L

#--------------------------------------------------------------------------------------------------
def editparams(d,usermsg=None):
  r"""
:param d: target dictionary
:type d: :class:`Dict[str,str]`

Edits a dictionary *d* and returns it. The editor can change the values but not the keys.
  """
#--------------------------------------------------------------------------------------------------
  p = {}
  L = list(d.keys())
  try:
    while True:
      print('\x1Bc',end='')
      if usermsg is not None: print(usermsg)
      for k,v in d.items(): print(k,'=',p.get(k,v))
      while True:
        choice = input('Ret: exit; -*: unset; *=*: set> ').strip()
        if choice=='': d.update(p); return d
        else:
          if choice[0] == '-':
            k = choice[1:]
            try: del p[k]
            except KeyError: print('invalid key'); continue
          else:
            try: k,v = choice.split('=',1)
            except: print('invalid syntax'); continue
            else:
              if k in L: p[k] = v
              else: print('invalid key'); continue
          break
  finally: print('\x1Bc',end='',flush=True)

#--------------------------------------------------------------------------------------------------
def checktype(name,x,*typs,allow_none=False):
#--------------------------------------------------------------------------------------------------
  if not ((allow_none and x is None) or isinstance(x,typs)):
    raise TypeError('{} must be of type {}, not {}'.format(name,'|'.join(typs),type(x)))
