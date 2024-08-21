import sys
import importlib

module_name=sys.argv[1].split('.py')[0]
module_name=module_name.replace('/','.')
print(module_name)
module=importlib.import_module(module_name)
module.main()
