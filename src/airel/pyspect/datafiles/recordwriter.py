import hashlib
import os.path

import numpy
import yaml


class SpectraWriterFile:
    def __init__(self, file_template, inverter_list, overwrite=False, append=False):
        self.file_template = file_template
        self.format_time = str
        self.overwrite = overwrite
        self.append = append
        self.files = {}

        if isinstance(inverter_list, dict):
            myinverters = []
            keys = [x for x in inverter_list.keys() if x is not None]
            keys.sort()
            if None in inverter_list:
                keys.append(None)
            self.inverter_list = [inverter_list[x] for x in keys]

        else:
            self.inverter_list = inverter_list

    def get_file(self, time):
        name = time.strftime(self.file_template)
        file = self.files.get(name, None)

        if file is not None:
            if file.closed:
                is_closed = True
            else:
                return file
        else:
            is_closed = False

        for oldfiles in self.files.values():
            oldfiles.close()

        if (
            not is_closed
            and os.path.exists(name)
            and not self.overwrite
            and not self.append
        ):
            raise IOError("File '%s' already exists. Not overwriting." % name)

        if is_closed or self.append:
            f = open(name, "a")
        else:
            f = open(name, "w")

        self.write_header(f)
        self.files[name] = f
        return f

    def write_header(self, file):
        header = {"spectra": []}
        columns = ["begin_time", "end_time", "opmode"]

        for i in self.inverter_list:
            header["spectra"].append(
                {
                    "name": i.name,
                    "yunit": i.yunit,
                    "xpoints": i.xpoints.tolist(),
                    "xunit": i.xunit,
                    "scale": i.xtype,
                }
            )
            colname = "".join(i.name.split())
            for dp in i.xpoints:
                columns.append("%s_%g" % (colname, dp))
            for dp in i.xpoints:
                columns.append("%s_err_%g" % (colname, dp))

        file.write("# Spectops spectra\n")
        for l in yaml.safe_dump(header, line_break="\n").split("\n"):
            file.write("# %s\n" % l)
        file.write("\t".join(columns))
        file.write("\n")

    def write_record(self, begintime, endtime, opmode, spectra, flush):

        columns = [self.format_time(begintime), self.format_time(endtime), opmode]
        for inv in self.inverter_list:
            if inv in spectra:
                sp = spectra[inv]

                for v in sp[0]:
                    if numpy.isnan(v):
                        columns.append("")
                    else:
                        columns.append(str(v))
                for v in sp[1]:
                    if numpy.isnan(v):
                        columns.append("")
                    else:
                        columns.append(str(v**0.5))
            else:
                columns += [""] * (len(inv.distribution_points) * 2)

        f = self.get_file(endtime)
        f.write("\t".join(columns))
        f.write("\n")
        if flush:
            f.flush()


class SpectraWriter:
    def __init__(self, file_template, invertermap, overwrite=False, append=False):
        self.invertermap = invertermap
        self.file_template = file_template
        self.overwrite = overwrite
        self.append = append

        self.outputfiles = {}

    def write_record(self, begintime, endtime, opmode, spectra, flush=False):
        try:
            of = self.outputfiles[opmode]
        except KeyError:
            if opmode not in self.invertermap:
                return
            filename = self.file_template.format(time="%Y%m%d", opmode=opmode)

            of = SpectraWriterFile(
                filename,
                self.invertermap[opmode],
                overwrite=self.overwrite,
                append=self.append,
            )
            self.outputfiles[opmode] = of

        of.write_record(begintime, endtime, opmode, spectra, flush=flush)


def format_time(dt):
    utcoffset = dt.utcoffset().total_seconds() / 60
    if utcoffset < 0:
        sign = "-"
        hours, minutes = divmod(-utcoffset, 60)
    else:
        sign = "+"
        hours, minutes = divmod(utcoffset, 60)
    return "{:%Y-%m-%d %H:%M:%S.%f}{}{:02d}:{:02d}".format(
        dt, sign, int(hours), int(minutes)
    )


class RecordWriter:
    def __init__(self, file_name_template, metadata):
        self.file_name_template = file_name_template
        self.metadata = metadata
        self.open_file_name = None
        self.output_file = None
        self.flag_map = {}

    def write_record(self, rec, flush=False):
        self.init_file(rec)

        flag_ids = [self.get_flag_id(f) for f in rec["flags"]]

        output = self.output_file

        output.write(rec["begin_time_str"])
        output.write("\t")
        output.write(rec["end_time_str"])
        output.write("\t")
        output.write(rec["opmode"])
        output.write("\t")

        for c in rec["current"]:
            output.write(str(c))
            output.write("\t")
        for c in rec["current_variance"]:
            output.write(str(c))
            output.write("\t")
        for c in rec["raw_current"]:
            output.write(str(c))
            output.write("\t")
        for c in rec["electrometer_voltage"]:
            output.write(str(c))
            output.write("\t")
        for c in rec["parameters"]:
            output.write(str(c))
            output.write("\t")

        for f in flag_ids:
            output.write("!")
            output.write(f)

        output.write("\n")

        if flush:
            self.output_file.flush()

    def init_file(self, rec):
        file_name = rec["begin_time"].strftime(self.file_name_template)
        if file_name == self.open_file_name:
            return

        print("Opening new file {}".format(file_name))
        self.open_file_name = file_name

        self.output_file = open(file_name, "a")
        self.flag_map = {}

        self.write_header()

    def write_header(self):
        output = self.output_file

        output.write("# Spectops records\n")

        header = {
            "file type": "records",
            "opmodes": [m["name"] for m in self.metadata["opmode_list"]],
            "electrometer names": self.metadata["electrometer_names"],
            "electrometer groups": self.metadata["electrometer_groups"],
            "parameters": [
                {"humanname": p["name"], "name": p["id"], "unit": p["unit"]}
                for p in self.metadata["measurement_parameters"]
            ],
            "total electrometers": len(self.metadata["electrometer_names"]),
        }

        for l in yaml.safe_dump(header, allow_unicode=True).split("\n"):
            output.write("# ")
            output.write(l)
            output.write("\n")

        parameter_ids = [p["id"] for p in self.metadata["measurement_parameters"]]
        num_electrometers = len(self.metadata["electrometer_names"])

        columns = (
            ["begin_time", "end_time", "opmode"]
            + ["cur_{}".format(i) for i in range(num_electrometers)]
            + ["curvar_{}".format(i) for i in range(num_electrometers)]
            + ["rawcur_{}".format(i) for i in range(num_electrometers)]
            + ["volt_{}".format(i) for i in range(num_electrometers)]
            + parameter_ids
            + ["flags"]
        )

        output.write("\t".join(columns))
        output.write("\n")

    def get_flag_id(self, flag):
        try:
            return self.flag_map[flag]
        except KeyError:
            fullfid = hashlib.sha1(flag.encode("utf8")).hexdigest()
            vals = self.flag_map.values()
            fid = fullfid[:2]
            i = 2
            while fid in vals:
                fid += fullfid[i % len(fullfid)]
                i += 1
            self.flag_map[flag] = fid
            self.write_flag_def(fid, flag)
            return fid

    def write_flag_def(self, fid, flag):
        self.output_file.write("# flag {}: {}\n".format(fid, flag.encode("utf8")))
        # sys.stdout.write('# flag {}: {}\n'.format(fid, flag.encode('utf8')))
