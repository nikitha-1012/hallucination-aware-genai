# Network Security Fundamentals

## Firewalls

A firewall is a network security device that monitors and controls incoming and outgoing network traffic based on predetermined security rules. Firewalls establish a barrier between a trusted internal network and an untrusted external network such as the Internet.

Common firewall types include:
- Packet Filtering Firewalls: Inspect packets at the network and transport layers, examining source and destination IP addresses, ports, and protocols. Packet filtering firewalls do not inspect payload contents or connection state.
- Stateful Inspection Firewalls: Track the operating state of active network connections (such as TCP handshakes) and determine whether packets belong to an established connection before allowing traffic.
- Application-Level Gateway / Proxy Firewalls: Operate at the application layer to inspect payload content for specific protocols such as HTTP and FTP, filtering malicious application-layer commands.

## Virtual Private Networks (VPN)

A Virtual Private Network creates an encrypted connection over an untrusted network. VPNs secure data transmission by encapsulating network packets within encrypted tunnels.

Key VPN protocols:
- IPsec (Internet Protocol Security): Operates at the network layer to authenticate and encrypt IP packets across network gateways.
- OpenVPN: Uses the OpenSSL library and TLS/SSL protocols to provide secure site-to-site or remote-access connections.

VPNs protect data confidentiality and integrity against eavesdropping and man-in-the-middle attacks on public networks.

## Intrusion Detection and Prevention Systems (IDS / IPS)

Intrusion Detection Systems (IDS) and Intrusion Prevention Systems (IPS) monitor network traffic for suspicious activity and known threats.

Key differences:
- IDS (Intrusion Detection System): Operates passively by monitoring traffic, analyzing signatures or behavioral anomalies, and generating alerts for system administrators without halting traffic directly.
- IPS (Intrusion Prevention System): Operates in-line with network traffic, actively dropping malicious packets, blocking offending IP addresses, and resetting unauthorized TCP connections in real time.

Detection methodologies include signature-based detection (comparing traffic against databases of known attack patterns) and anomaly-based detection (identifying deviations from baseline normal network behavior).

## Port Scanning

Port scanning is the process of probing a target host's transport layer ports to discover active listening services and potential vulnerabilities.

Common scan techniques:
- TCP SYN Scan (Half-Open Scan): The scanner sends a SYN packet; if the target responds with SYN-ACK, the port is open; the scanner resets the connection with RST before completing the three-way handshake.
- TCP Connect Scan: Completes the full three-way TCP handshake (SYN, SYN-ACK, ACK), making it reliable but easily logged by firewalls and intrusion detection systems.
