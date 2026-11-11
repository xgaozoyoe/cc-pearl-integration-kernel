"""Map nvidia-smi PCI address to Nsight's own metrics-device enumeration."""
import re,sys
from pathlib import Path

def pci_key(value):
    match=re.search(r'([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-7])',value)
    if not match:raise ValueError('Invalid PCI bus address: '+value)
    return tuple(int(x,16) for x in match.groups())

def select_device(text, pci):
    matches=[]
    for line in text.splitlines():
        bus=re.search(r'PCI\[([^]]+)\]',line,re.I)
        if not bus or pci_key(bus[1])!=pci_key(pci):continue
        # Supported-device list uses a numeric ID prefix. Never infer an ID
        # from line order in the unsupported/permission-error list.
        prefix=line[:bus.start()]
        number=re.match(r'^\s*(?:(?:GPU|Device)\s*)?\[?(\d+)\]?\s*[:=.-]\s*',prefix,re.I)
        if number and 'Insufficient privilege' not in line:matches.append(int(number[1]))
    if len(matches)!=1:
        raise ValueError('Cannot uniquely map PCI address to a supported Nsight ID. Inspect device-list.txt; refusing capture.')
    return matches[0]

if __name__=='__main__':
    print(select_device(Path(sys.argv[1]).read_text(),sys.argv[2]))
