start_time = time.localtime()
runstring = time.strftime("%Y-%m-%d-%a-%H-%M-%S", start_time)
self.filename = "%s-%s.hdf5" % (config["General"]["FileNamePrefix"],
                                runstring)