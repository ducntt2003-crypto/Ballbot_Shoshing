from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sloshbot.experiment import main
if __name__=='__main__':
    if '--controller' in sys.argv:raise SystemExit('Teleoperation exposes the FF/Heading adapter; use --modal-feedback only for its declared privileged diagnostic. Other controller interfaces are not certified.')
    main()
