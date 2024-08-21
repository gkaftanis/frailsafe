import importlib

registered_plugins={\
        'coinbase_pro':'coinbase_pro_tracker',
        'oanda':'oanda_tracker'
}

class Registry(object):
    @classmethod
    def get_class_for_plugin(cls,name):
        module=importlib.import_module('.'+registered_plugins[name],
                                      package='participants_source.plugins')
        return module.get_plugin_class()
