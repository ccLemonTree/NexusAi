# -*- coding: utf-8 -*-
"""
@author chenhaolin
@date 2023年03月17日 10:57:14
@packageName Triton
@className singel
@version 1.0.0
@describe TODO
"""
def Singleton(cls):
	_instance={}
	def _singleton(*args,**kwagrs):
		if cls not in  _instance:
			_instance[cls]=cls(*args,**kwagrs)
		return _instance[cls]
	return _singleton