export let attackData = {
    "Reconnaissance": [],
    "Resource Development": [
        { "technique": "T1587.001 - Develop Capabilities: Malware", "description": "The threat actor developed a custom Python information stealer." },
        { "technique": "T1608.001 - Stage Capabilities: Upload Malware", "description": "The threat actor staged malware on DropBox." },
        { "technique": "T1608.005 - Stage Capabilities: Link Target", "description": "The threat actor distributed links to DropBox files." },
    ],
    "Initial Access": [
        { "technique": "T1566.002 - Phishing: Spearphishing Link", "description": "The threat actor distributed links to DropBox files." },
        { "technique": "T1566.003 - Phishing: Phishing via Service", "description": "The threat actor used Skype and Google chats for phishing." },
    ],
    "Execution": [
        { "technique": "T1204.002 - User Execution: Malicious File", "description": "The threat actor's payload relied on manual user execution." },
        { "technique": "T1059.005 - Command and Scripting Interpreter: Python", "description": "The threat actor used a packaged Python interpreter to execute code." },
    ],
    "Persistence": [
        { "technique": "T1547.001 - Boot or Logon Autostart Execution: Registry Run Keys / Startup Folder", "description": "The threat actor created a Run key." },
        { "technique": "T1053.005 - Scheduled Task/Job: Scheduled Task", "description": "The threat actor created a scheduled task." },
    ],
    "Privilege Escalation": [
    ],
    "Defense Evasion": [
        { "technique": "T1036.008 - Masquerading: Masquerade File Type", "description": "The malware executable used a PDF file icon." },
        { "technique": "T1553.005 - Subvert Trust Controls: Mark-of-the-Web Bypass", "description": "The threat actor avoided MOTW distributing the malware using a protected archive." },
        { "technique": "T1211 - Exploitation for Defense Evasion", "description": "First stage malware contained a SmartScreen bypass." },
        { "technique": "T1027.001 - Obfuscated Files or Information: Binary Padding", "description": "First stage malware used nonstandard binary padding." },
        { "technique": "T1027.009 - Obfuscated Files or Information: Embedded Payloads", "description": "First stage malware contained second stage, decoy files and python interpreter." },
        { "technique": "T1027.013 - Obfuscated Files or Information: Encrypted/Encoded File", "description": "Final payload decrypted and executed by stub." },

    ],
    "Credential Access": [
        { "technique": "T1555.003 - Credentials from Password Stores: Credentials from Web Browsers", "description": "Malware obtained credentials from browser stores." },
        { "technique": "T1539 - Steal Web Session Cookie", "description": "Malware used CDP and remote debugging to obtain cookies." },
    ],
    "Discovery": [
        { "technique": "T1016 -System Network Configuration Discovery", "description": "Malware used ipinfo.io service to obtain external IP." },
        { "technique": "T1057 - Process Discovery", "description": "Malware discovered running processes with tasklist." },

    ],
    "Lateral Movement": [],
    "Collection": [
        { "technique": "T1560 - Archive Collected Data", "description": "Malware packaged exfiltrated data into a ZIP archive." },
        { "technique": "T1005 - Data from Local System", "description": "Malware collects PDF, txt and wallet files from local disk." },
        { "technique": "T1113 - Screen Capture", "description": "Malware captures a screenshot on execution." },

    ],
    "Command and Control": [
        { "technique": "T1102 - Web Service", "description": "Malware executed commands from remote webpage." },
    ],
    "Exfiltration": [
        { "technique": "T1567 - Exfiltration Over Web Service", "description": "Malware exfiltrates data using Telegram bot API." },
    ],
    "Impact": [
        { "technique": "T1657 - Financial Theft", "description": "Malware rewrites clipboard contents to steal cryptocurrency." },
        { "technique": "T1496 - Resource Hijacking", "description": "Malware targets TikTok and Meta advertising cookies to abuse ad accounts." },
    ]
};