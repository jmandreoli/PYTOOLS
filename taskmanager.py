# File:                 taskmanager.py
# Creation date:        2013-11-18
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Simple task utilities
#
# *** Copyright (c) 2012 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

import logging, os, re, icalendar, uuid, unicodedata, sys
from datetime import datetime, timedelta
from copy import deepcopy
from pytz import timezone, utc
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from smtplib import SMTP
from lxml.etree import parse as xmlparse, tostring as xmltostring, tounicode as xmltounicode, iselement as isxmlelement
logger = logging.getLogger(__name__)

#------------------------------------------------------------------------------
# def mailprepare
#------------------------------------------------------------------------------

def mailprepare(msg=None,fromaddr=None,toaddr=None,ccaddr=None,bccaddr=None,dstarget=None):
  r"""
:param msg: email message
:type msg: :class:`email.message`
:param fromaddr: email address for 'from' field
:type fromaddr: :const:`str`
:param toaddr: list of email addresses for 'to' field
:type toaddr: list(:const:`str`) | :const:`NoneType`
:param ccaddr: list of email addresses for 'cc' field
:type ccaddr: list(:const:`str`) | :const:`NoneType`
:param bccaddr: list of email addresses for 'bcc' field
:type bccaddr: list(:const:`str`) | :const:`NoneType`
:param dstarget: docushare collection identifier for subject extension
:type dstarget: :const:`str` | :const:`NoneType`

Appends to,cc,bcc fields to the email message *msg*, and possibly extends subject for docushare agent interaction.
  """
  msg.add_header('from',fromaddr)
  sender = fromaddr
  recipient = []
  for hdr,addrs in (('to',toaddr),('cc',ccaddr),('bcc',bccaddr)):
    if addrs is not None:
      msg.add_header(hdr,', '.join(addrs))
      recipient.extend(addrs)
  if dstarget:
    msg.replace_header('subject','{0} <upload: {1}>'.format(msg['subject'],dstarget))
  return dict(sender=sender,recipient=set(recipient),message=msg)

#------------------------------------------------------------------------------
# def maildisplay
#------------------------------------------------------------------------------

def maildisplay(message,textshow=None):
  r"""
:param message: email message 
:type message: :class:`email.message`

Displays content of *message*.
  """
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
  L = tuple(struct(message))
  if L:
    print()
    for i,msg in enumerate(L,1):
      print('<<<------------ Content of part ',i,'------------>>>')
      content = msg.get_payload(decode=True)
      print(content.decode(str(msg.get_charset())))

#------------------------------------------------------------------------------
# def mailsend
#------------------------------------------------------------------------------

def mailsend(sender=None,recipient=None,message=None,mailhost='smtphost'):
  r"""
:param sender: specification of sender (unique email address)
:type sender: :const:`str`
:param recipient: specification of recipients (comma separated email addresses)
:type recipient: :const:`str`
:param message: message content
:type message: :class:`email.message`

Sends *message* as email on *mailhost* with specified *sender* and *recipient*.
  """
  server = SMTP(mailhost)
  try: err = server.sendmail(sender,recipient,message.as_string())
  finally: server.quit()
  if err: raise Exception('At least one recipient did not receive the mail',err)

#------------------------------------------------------------------------------
# def announcement
#------------------------------------------------------------------------------

def announcement(shorttxt=None,plaintxt=None,xhtml=None,icscal=None,filename=None,charset='utf-8'):
  r"""
:param icscal: calendar version of the announcement
:type icscal: :class:`icalendar.Calendar`
:param plaintxt: plain text version of the announcement
:type plaintxt: :const:`str`
:param xhtml: xhtml version of the announcement
:type xhtml: :class:`lxml.etree._Document`
:param shorttxt: short (1 liner) text version of the announcement
:type shorttxt: :const:`str`
:param filename: file name associated to the ics calendar in the announcement
:type filename: :const:`str`
:param charset: charset encoding specification
:type charset: :const:`str`
:rtype: :class:`email.message`

Returns the announcement specified in different forms (*shorttxt*, *plaintxt*, *xhtml*, *icscal*) as an email message.
  """
  assert isinstance(shorttxt,str)
  assert isinstance(plaintxt,str)
  assert hasattr(xhtml,'docinfo') and hasattr(xhtml,'getroot')
  assert isxmlelement(xhtml.getroot())
  assert isinstance(icscal,icalendar.Calendar)
  msg = MIMEMultipart('alternative')
  msg.set_param('name','announce.eml')
  msg.add_header('subject',shorttxt)
  msg.attach(MIMEText(plaintxt,'plain',charset))
  msg.attach(MIMEText(xmltounicode(xhtml),'html',charset))
  m = MIMEText(icscal.to_ical().decode('utf-8'),'calendar',charset)
  m.set_param('method',icscal.get('method'))
  if filename is not None: m.add_header('content-disposition','attachment',filename=filename+'.ics')
  msg.attach(m)
  return msg

#------------------------------------------------------------------------------
# def calendar
#------------------------------------------------------------------------------

def calendar(content=None,method='PUBLISH',version='2.0',reminder=None):
  r"""
:param content: list of events
:type content: list(:class:`icalendar.Event`)
:param reminder: reminder
:type reminder: :class:`datetime.timedelta`
:param method: ics method for the icalendar
:type method: :const:`str`
:param version: icalendar version;should not be changed
:type version: :const:`str`
:rtype: :class:`icalendar.Calendar`

Return a calendar object listing all the events in *content*, possibly augmented, for those confirmed, with a reminder specified by *reminder* (from start).
  """
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

#------------------------------------------------------------------------------
# def event
#------------------------------------------------------------------------------

def event(start=None,duration=None,permalink=None,sequence=0,confirmed=True,priority=5,klass='PUBLIC',transp='OPAQUE'):
  r"""
:param start: event start instant, in UTC
:type start: :class:`datetime.datetime`
:param duration: event duration
:type duration: :class:`datetime.timedelta`
:param confirmed: whether the event is confirmed
:type confirmed: :const:`bool`
:param permalink: permanent url for the event
:type permalink: :const:`str` | :const:`NoneType`
:rtype: :class:`icalendar.Event`

Returns a calendar event, which can be further extended. The *permalink*, if not const:`None`, is used to generate a unique ID of the event.
  """
  assert isinstance(start,datetime)
  start = start.astimezone(utc)
  assert isinstance(duration,timedelta)
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

#------------------------------------------------------------------------------
# class Transaction
#------------------------------------------------------------------------------

class Transaction (object):
  r"""
Instances of this class specify a (very simple) transaction, consisting of a prepare phase and a commit phase. A transaction can have subtransactions.

Attributes:

.. attribute:: subt

   list of subtransactions

.. attribute:: executed

   :const:`bool` flag indicating whether the trasaction has been executed.
   A transaction can only be executed only once.

Methods:
  """

  def __init__(self):
    self.subt = []
    self.executed = False

  def __call__(self,dryrun=False):
    assert not self.executed
    self.executed = True
    logger.info('Transaction[%s]: started%s',self,(' (dry run mode)' if dryrun else ''))
    try: plan = tuple(self.prepareall())
    except KeyboardInterrupt:
      logger.info('Transaction[%s]: aborted',self)
      return
    plan = reversed(plan)
    if dryrun: print('Plan', tuple(plan))
    else:
      for t in plan: t.commit()

  def prepareall(self):
    self.prepare()
    yield self
    for t in self.subt:
      for tt in t.prepareall(): yield tt

  def prepare(self):
    r"""
Prepares step execution.
    """
    pass

  def commit(self):
    r"""
Commits step execution.
    """
    logger.info('Transaction[%s]: committed',self)

  def addtr(self,t):
    r"""
:param t: transaction object
:type t: :class:`Transaction`

Adds transaction *t* as subtransaction to *self*.
    """
    assert isinstance(t,Transaction)
    self.subt.append(t)

#------------------------------------------------------------------------------
# class MailingTransaction
#------------------------------------------------------------------------------

class MailingTransaction (Transaction):
  r"""
A mailing transaction. On prepare: compute sender, recipient and ask for confirmation; on commit: perform the sending.

:param mailhost: specification of the mail host
:type mailhost: :const:`str`
:param mailmsg: message to send
:type mailmsg: :class:`email.message`
:param distrib: specification of 'from','to','cc','bcc' fields
:type distrib: :const:`dict`
  """

  def __init__(self,mailhost=None,mailmsg=None,distrib=None):
    self.mailhost = mailhost
    self.mailmsg = mailmsg
    self.distrib = distrib
    super(MailingTransaction,self).__init__()

  def prepare(self):
    self.mail = mailprepare(self.mailmsg,**self.distrib)
    self.getconfirm()
    super(MailingTransaction,self).prepare()

  def commit(self):
    mailsend(mailhost=self.mailhost,**self.mail)
    super(MailingTransaction,self).commit()

  def getconfirm(self,LINE=79*'-',textshow=('plain',)):
    print('Confirm sending the following mail:')
    print(LINE)
    maildisplay(self.mailmsg,textshow)
    print(LINE)
    confirm()

#------------------------------------------------------------------------------
# class XMLFileTransaction
#------------------------------------------------------------------------------

class XMLFileTransaction (Transaction):
  r"""
An XML file transaction. On prepare: parse the xml file and select nodes in the document; on commit: save the document.

:param path: specification of the xml file path
:type path: :const:`str`
:param namespaces: association of xml tag prefixes with names
:type namespaces: :const:`dict`
:param target: xpath specification of initial selection
:type target: :const:`str`
:param targetnamer: xpath specification to name a selected node
:type targetnamer: :const:`str`
  """

  def __init__(self,path=None,namespaces=None,target=None,targetnamer=None):
    self.path = path
    self.namespaces = namespaces
    self.target = target
    self.targetnamer = targetnamer
    self.unmodified = False
    super(XMLFileTransaction,self).__init__()

  def prepare(self):
    self.doc = xmlparse(self.path)
    self.select()
    super(XMLFileTransaction,self).prepare()

  def commit(self):
    super(XMLFileTransaction,self).commit()
    if self.unmodified: return
    safedump(xmltostring(self.doc,encoding=self.doc.docinfo.encoding,xml_declaration=True),self.path,'wb')

  def select(self):
    self.target = self.doc.xpath(self.target,namespaces=self.namespaces)

  def select1(self):
    self.target = choose1(self.target,pname=lambda e: '|'.join(e.xpath(self.targetnamer,namespaces=self.namespaces)))

#------------------------------------------------------------------------------
# Utilities
#------------------------------------------------------------------------------

def xmlpuretext(e,pat=re.compile('(\n{2,})',re.UNICODE)):
  r"""
Returns the text content of *e*.

:param e: XML node
:type e: :class:`lxml.etree._Element`
  """
  return pat.sub('\n',''.join(e.xpath('descendant-or-self::text()')))

def xmlsubstitute(doc,d):
  r"""
Replaces each node of the form <?parm xx?> in *doc* by *d* ['xx'], which must be an XML node.

:param doc: target xml document
:type doc: :class:`lxml.etree._Element`
:param d: substitution table
:type d: :const:`dict`
  """
  for parm in doc.xpath('//processing-instruction("parm")'):
    x = d.get(parm.text)
    if x is not None:
      x = deepcopy(x)
      x.tail = parm.tail
      parm.getparent().replace(parm,x)
  return doc

def unicodetoascii(x,pat=re.compile(' WITH (?:ACUTE|GRAVE|CIRCUMFLEX|DIAERESIS|CEDILLA|TILDE|STROKE:)')):
  r"""
Returns an ascii version of string *x*.

:param x: target string
:type x: :const:`str`
  """
  return str(''.join(unicodedata.lookup(pat.sub('',unicodedata.name(c))) for c in x))

def safedump(content,path,mode='w'):
  r"""
Saves string *content* into filesystem member *path* with some safety.

:param content: target string
:type content: :const:`str`
:param path: target path
:type path: :const:`str`
  """
  if os.path.exists(path):
    dirn,fn = os.path.split(path)
    pathnew,pathsav = os.path.join(dirn,'new-'+fn),os.path.join(dirn,'bak-'+fn)
    with open(pathnew,mode) as v: v.write(content)
    if os.path.exists(pathsav): os.remove(pathsav)
    os.rename(path,pathsav)
    os.rename(pathnew,path)
  else:
    with open(path,mode) as v: v.write(content)

def choose1(L,pname,LINE=79*'-'):
  r"""
Picks option from menu *L*.

:param L: list of options
:type L: list(:const:`object`)
:param pname: print-name function for the options
:type pname: :const:`object` -> :const:`str`
  """
  N = len(L)
  assert N>0, Exception('No entry to choose from')
  if N > 1:
    print(LINE)
    for j,e in enumerate(L,1):
      print('({0}) <|{1}|>'.format(j,pname(e)))
    print(LINE)
    while True:
      try: choice = input('Select? [1-%d: selection]>>> '%N)
      except KeyboardInterrupt: print(); raise
      try:
        j = int(choice,10)
        if j < 1 or j > N: raise Exception('Out of bounds')
        j -= 1
        break
      except: pass
  else:
    j = 0
  return L[j]

def editparams(d,LINE=79*'-'):
  r"""
Edits a dictionary *d*.

:param d: target dictionary
:type d: :const:`dict`
  """
  p = {}
  L = list(d.keys())
  L.sort()
  while True:
    try: choice = input('Select? [Ret: save; ?: display; -*: del; {%s}=*: set]>>> '%'|'.join(L))
    except KeyboardInterrupt: print(); raise
    if choice == '': break
    elif choice == '?':
      print(LINE)
      for k in L:
        print(k, '=', p.get(k,d[k]))
    elif choice[0] == '-':
      k = choice[1:]
      if p.has_key(k): del p[k]
      else: print('ignored')
    else:
      try: k,v = choice.split('=',1)
      except: print('ignored')
      else:
        if k not in L: continue
        p[k] = v
  d.update(p)

def confirm():
  r"""
Requests a confirmation.
  """
  while True:
    try: choice = input('[Ret: confirm]>>> ')
    except KeyboardInterrupt: print(); raise
    if choice=='': return
    else: print('ignored')
