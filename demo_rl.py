import sys
from sloshbot.experiment import main
if __name__=='__main__':
    if '--controller' not in sys.argv:sys.argv.extend(['--controller','PPO'])
    main()
