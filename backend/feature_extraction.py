import re
from urllib.parse import urlparse

SUSPICIOUS_TLDS={"xyz","top","club","info","tk","ml","ga","cf","gq","work","click","loan","win","review","download","zip"}
SHORTENERS={"bit.ly","tinyurl.com","t.co","goo.gl","ow.ly","is.gd","cutt.ly"}
BRANDS={"paypal","amazon","apple","microsoft","netflix","bank","google","facebook","instagram","whatsapp","sbi","hdfc","icici","paytm"}
URL_WORDS={"login","verify","secure","account","update","confirm","signin","password","unlock","suspend","billing","invoice","reset","wallet"}
URGENCY={"urgent","immediately","act now","expire","expires","suspend","verify now","limited time","action required","asap"}
REWARDS={"winner","congratulations","claim","free","prize","gift card","lottery","jackpot","cashback","reward","bonus"}
SENSITIVE={"otp","password","pin","cvv","ssn","aadhar","card number","account number","login details","bank details"}
GREETINGS={"dear customer","dear user","dear valued customer","dear account holder","dear sir/madam"}
IP_RE=re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
URL_RE=re.compile(r"(?:https?://\S+|www\.\S+)",re.I)

def extract_url_features(url):
    s=url.strip(); target=s if re.match(r"^[A-Za-z]+://",s) else "http://"+s
    p=urlparse(target); host=p.hostname or ""; low=s.lower(); reasons=[]
    dots=host.count('.'); subs=max(dots-1,0); hyphens=host.count('-'); digits=sum(c.isdigit() for c in host)
    special=sum(s.count(c) for c in ['@','%','=','&','//']); at='@' in s; ip=bool(IP_RE.match(host))
    https=p.scheme.lower()=='https'; tld=host.split('.')[-1] if '.' in host else ''
    sus=tld in SUSPICIOUS_TLDS; short=host in SHORTENERS
    sens=[w for w in URL_WORDS if w in low]; brands=[b for b in BRANDS if b in host.replace('-','')]
    port=p.port is not None; path=p.path or ''; q=p.query or ''; params=len([x for x in q.split('&') if x]); dslash='//' in path+q
    if len(s)>75: reasons.append(f'Unusually long URL ({len(s)} characters)')
    if subs>=3: reasons.append(f'Excessive number of subdomains ({subs})')
    if hyphens>=2: reasons.append('Multiple hyphens in domain name')
    if digits>=4: reasons.append('Domain contains many digits')
    if at: reasons.append("URL contains '@'")
    if ip: reasons.append('Domain is a raw IP address')
    if not https: reasons.append('Connection is not secured with HTTPS')
    if sus: reasons.append(f'Uses suspicious TLD (.{tld})')
    if short: reasons.append('URL uses a link-shortening service')
    if sens: reasons.append('URL contains sensitive/action keywords: '+', '.join(sorted(sens)[:4]))
    if brands: reasons.append('Domain references a known brand: '+', '.join(brands))
    if port: reasons.append('URL specifies a port')
    if dslash: reasons.append('Contains redirect-like double slash')
    if not reasons: reasons.append('No strong red flags detected in URL structure')
    raw={'url_length':len(s),'num_dots':dots,'num_subdomains':subs,'num_hyphens':hyphens,'num_digits_in_host':digits,'num_special_chars':special,'has_at_symbol':at,'is_ip_address':ip,'has_https':https,'tld':tld,'suspicious_tld':sus,'is_shortener':short,'sensitive_word_hits':sens,'brand_mentions':brands,'has_port':port,'path_length':len(path),'num_query_params':params,'double_slash_redirect':dslash}
    vec=[len(s),dots,subs,hyphens,digits,special,int(at),int(ip),int(not https),int(sus),int(short),len(sens),len(brands),int(port),len(path),params,int(dslash)]
    return vec,reasons,raw

URL_FEATURE_NAMES=['url_length','num_dots','num_subdomains','num_hyphens','num_digits','num_special_chars','has_at_symbol','is_ip_address','missing_https','suspicious_tld','is_shortener','sensitive_word_count','brand_mention_count','has_port','path_length','num_query_params','double_slash_redirect']

def extract_message_features(text):
    s=text.strip(); low=s.lower(); reasons=[]; urls=URL_RE.findall(s)
    urgency=[w for w in URGENCY if w in low]; rewards=[w for w in REWARDS if w in low]; sensitive=[w for w in SENSITIVE if w in low]; greetings=[w for w in GREETINGS if w in low]; brands=[w for w in BRANDS if w in low]
    ex=s.count('!'); words=re.findall(r'[A-Za-z]+',s); upper=[w for w in words if len(w)>2 and w.isupper()]; money=bool(re.search(r'[$₹€£]\s?\d|amount of|lottery|million|lakh|crore',low))
    if urls: reasons.append(f'Message contains {len(urls)} embedded link(s)')
    if urgency: reasons.append('Uses urgency/pressure language')
    if rewards: reasons.append('Promises rewards/prizes')
    if sensitive: reasons.append('Requests sensitive information')
    if greetings: reasons.append('Uses a generic greeting')
    if ex>=3: reasons.append('Excessive exclamation marks')
    if len(upper)>=3: reasons.append('Excessive ALL CAPS words')
    if money: reasons.append('Mentions money or winnings')
    if brands: reasons.append('References a well-known brand')
    if not reasons: reasons.append('No strong red flags detected in message content')
    raw={'msg_length':len(s),'num_urls':len(urls),'urgency_hits':urgency,'reward_hits':rewards,'sensitive_info_hits':sensitive,'generic_greeting':bool(greetings),'num_exclamations':ex,'num_uppercase_words':len(upper),'money_mention':money,'brand_mentions':brands}
    return [len(s),len(urls),len(urgency),len(rewards),len(sensitive),int(bool(greetings)),ex,len(upper),int(money),len(brands)],reasons,raw

MESSAGE_FEATURE_NAMES=['msg_length','num_urls','urgency_word_count','reward_word_count','sensitive_info_count','generic_greeting','num_exclamations','num_uppercase_words','money_mention','brand_mention_count']
