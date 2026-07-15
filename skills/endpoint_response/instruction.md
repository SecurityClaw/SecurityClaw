# Endpoint Response

Prepare or execute defensive host actions on Linux and Windows. Every action requires a one-time authorization token echoed by the operator in the current message. Never claim success unless the result reports `status: ok`.

Supported operations include terminating a process, blocking a remote IP, removing a specific
ARP/NDP neighbor entry, and quarantining a file. Neighbor eviction is temporary and should be
paired with investigation of gateway, switch, DHCP, and endpoint evidence.
