import dataclasses
import datetime
import hashlib
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import IO, Any, Callable, Union

import numpy as np
import yaml

from .util import float_or_missing, parse_spectops_time
from ..fractions import nd_fraction_matrix

try:
    import pandas as pd
except ModuleNotFoundError:
    warnings.warn('Pandas not found. Dataframe conversion not available.')

nan = float('nan')


class FieldType(Enum):
    NORMAL = 0
    ELECTROMETER_CURRENT = 1
    ELECTROMETER_RAW_CURRENT = 2
    ELECTROMETER_VARIANCE = 3
    ELECTROMETER_VOLTAGE = 4


class SpectrumScale(Enum):
    DIAMETER = 1
    MOBILITY = 2


@dataclass
class Field:
    name: str
    human_name: str
    unit: str
    size: int
    field_type: FieldType = FieldType.NORMAL


@dataclass
class SpectraData:
    name: str
    inverter_name: str
    hash: str
    xunit: str
    yunit: str
    xpoints: list[float]
    scale: SpectrumScale
    xsize: int
    begin_time: list[datetime.datetime] = dataclasses.field(default_factory=list)
    end_time: list[datetime.datetime] = dataclasses.field(default_factory=list)
    value: list[list[float]] = dataclasses.field(default_factory=list)
    variance: list[list[float]] = dataclasses.field(default_factory=list)
    opmode: list[str] = dataclasses.field(default_factory=list)

    def fraction_concentration(self, limits):
        matrix = nd_fraction_matrix(xpoints=self.xpoints, limits=limits)
        fract_data = np.dot(self.value, matrix)

        return fract_data


class RecordsFiles:
    auto_parse_time: bool
    begin_time_str: list[str]
    end_time_str: list[str]
    size: int
    begin_time: Union[None, list[datetime.datetime]]
    end_time: Union[None, list[datetime.datetime]]
    opmode: list[str]
    field_data: dict[str, list[float]]
    current: list[list[float]]
    current_variance: list[list[float]]
    raw_current: list[list[float]]
    electrometer_voltage: list[list[float]]
    flags: list[list[int]]
    opmode_set: set[str]
    field_ids: list[str]
    flag_defs: list[str]
    field_definitions: dict[str, Field]
    num_electrometers: Union[None, int]
    electrometer_names: Union[None, list[str]]
    electrometer_groups: Union[None, dict[str, list[int]]]
    spectra: list[SpectraData]

    def __init__(self, auto_parse_time=True):
        self.size = 0
        self.begin_time_str = []
        self.end_time_str = []
        self.auto_parse_time = auto_parse_time
        if auto_parse_time:
            self.begin_time = []
            self.end_time = []
        else:
            self.begin_time = None
            self.end_time = None
        self.opmode = []
        self.field_data = {}
        self.current = []
        self.current_variance = []
        self.raw_current = []
        self.electrometer_voltage = []
        self.flags = []
        self.opmode_set = set()
        self.field_ids = []

        self.flag_defs = []
        self.field_definitions = {}
        self.num_electrometers = None
        self.electrometer_names = None
        self.electrometer_groups = None

        self.spectra = []

        self.warnings = []

    def load(self, sourcefile: Union[str, IO[str]]):
        is_in_header = False
        header_lines: list[str] = []
        linereader: Union[Callable[[int, str], None], None] = None

        if isinstance(sourcefile, str):
            sourcefile = open(sourcefile)

        for line_number, line in enumerate(sourcefile):
            if line[0] == '#':
                if line.startswith("# Spectops") or line.startswith("#Only records") or line.startswith(
                        "#All measured"):
                    is_in_header = True
                    header_lines = []

                if is_in_header:
                    header_lines.append(line)
                    continue
            elif is_in_header:
                is_in_header = False

                if line.startswith("begintime") or line.startswith("begin_time"):
                    header_lines.append(line)
                else:
                    self.warnings.append((sourcefile, line_number, 'Header ended unexpectedly'))
                    header_lines.append("")

                linereader = self.make_reader(header_lines)
                header_lines = []
                continue

            if linereader is not None:
                linereader(line_number, line)

    def count(self) -> int:
        return len(self.begin_time_str)

    def make_reader(self, lines: list[str]) -> Union[Callable[[int, str], None], None]:
        if lines[0] not in ["# Spectops records\n", "# Spectops spectra\n"]:
            raise ParsingError(f'Unknown file: {lines}')

        yamldoc = "".join(x[2:] for x in lines[1:-1])

        header = yaml.safe_load(yamldoc)
        file_type = header['file type']

        if file_type == 'records':
            return self.make_records_reader(header)
        elif file_type == 'spectra':
            return self.make_spectra_reader(header)
        else:
            return None

    def add_field(self, name, human_name, unit):
        field = Field(
            size=1,
            name=name,
            human_name=human_name,
            unit=unit,
        )
        self.field_definitions[name] = field
        data = [np.nan] * self.size
        self.field_data[name] = data
        self.field_ids.append(name)
        return data

    def make_records_reader(self, header: dict[str, Any]) -> Union[Callable[[int, str], None], None]:
        flag_map: dict[str, int] = {}

        self.opmode_set.update(header['opmodes'])

        if self.electrometer_names is None:
            self.electrometer_names = header.get('electrometer names', [])
            self.num_electrometers = len(self.electrometer_names)
            self.electrometer_groups = header.get('electrometer groups', {})
        else:
            if header['electrometer names'] != self.electrometer_names:
                raise ParsingError('Electrometer names do not match')

        num_electrometers = self.num_electrometers

        missing_fields = set(self.field_ids)
        file_field_data = []
        for field_dict in header['parameters']:
            name = field_dict['name']
            missing_fields.discard(name)
            if name not in self.field_definitions:
                data = self.add_field(name=name, human_name=field_dict['humanname'], unit=field_dict['unit'])
            else:
                data = self.field_data[name]
            file_field_data.append(data)

        missing_field_data = [self.field_data[name] for name in missing_fields]
        num_parameters = len(file_field_data)
        row_size = len(file_field_data) + num_electrometers * 4 + 4

        def reader(line_number: int, line: str):
            if line[0] == '#':
                if line.startswith('# flag'):
                    fid, part, flag = line[7:].rstrip().partition(': ')
                    try:
                        index = self.flag_defs.index(flag)
                    except ValueError:
                        index = len(self.flag_defs)
                        self.flag_defs.append(flag)
                    flag_map[fid] = index
            else:
                row_fields = line.rstrip('\n\r').split('\t')
                if len(row_fields) != row_size:
                    raise ParsingError(
                        "Invalid number of fields on row {}. Found {}, expected {}.".format(line_number + 1,
                                                                                            len(row_fields), row_size))
                self.begin_time_str.append(row_fields[0])
                self.end_time_str.append(row_fields[1])
                if self.auto_parse_time:
                    self.begin_time.append(parse_spectops_time(row_fields[0]))
                    self.end_time.append(parse_spectops_time(row_fields[1]))
                self.opmode.append(row_fields[2])
                fieldpos = 3
                self.current.append([float_or_missing(x) for x in row_fields[fieldpos:fieldpos + num_electrometers]])
                fieldpos += num_electrometers
                self.current_variance.append(
                    [float_or_missing(x) for x in row_fields[fieldpos:fieldpos + num_electrometers]])
                fieldpos += num_electrometers
                self.raw_current.append(
                    [float_or_missing(x) for x in row_fields[fieldpos:fieldpos + num_electrometers]])
                fieldpos += num_electrometers
                self.electrometer_voltage.append(
                    [float_or_missing(x) for x in row_fields[fieldpos:fieldpos + num_electrometers]])
                fieldpos += num_electrometers
                for i in range(num_parameters):
                    file_field_data[i].append(float_or_missing(row_fields[fieldpos + i]))
                fieldpos += num_parameters
                self.flags.append([flag_map[f] for f in row_fields[fieldpos].split('!') if f])
                for l in missing_field_data:
                    l.append(nan)

        return reader

    def make_spectra_reader(self, header: dict[str, Any]) -> Union[Callable[[int, str], None], None]:
        self.opmode_set.update(header['opmodes'])

        spectra = []

        for spectra_info in header['spectra']:
            spectrum_hash = spectra_info['hash']
            for s in self.spectra:
                if s.hash == spectrum_hash:
                    spectra.append(s)
                    break
            else:
                spectrum_data = init_spectrum_data(spectra_info)
                spectra.append(spectrum_data)
                self.spectra.append(spectrum_data)

        row_size = 3 + sum(s.xsize * 2 for s in spectra)

        def reader(line_number: int, line: str):
            if line[0] == '#':
                return
            else:
                row_fields = line.rstrip('\n\r').split('\t')
                if len(row_fields) != row_size:
                    raise ParsingError(
                        "Invalid number of fields on row {}. Found {}, expected {}.".format(line_number + 1,
                                                                                            len(row_fields), row_size))
                begin_time = parse_spectops_time(row_fields[0])
                end_time = parse_spectops_time(row_fields[1])
                opmode = row_fields[2]
                fieldpos = 3
                for s in spectra:
                    value = [float_or_missing(x) for x in row_fields[fieldpos:fieldpos + s.xsize]]
                    fieldpos += s.xsize
                    variance = [float_or_missing(x) for x in row_fields[fieldpos:fieldpos + s.xsize]]
                    fieldpos += s.xsize

                    s.opmode.append(opmode)
                    s.begin_time.append(begin_time)
                    s.end_time.append(end_time)
                    s.value.append(value)
                    s.variance.append(variance)

        return reader

    def sort(self):
        """Sort records

        Sort records based on begin_time. Raises RuntimeError when begin_time not parsed."""

        attrlist = ['begin_time_str', 'end_time_str', 'begin_time', 'end_time', 'opmode', 'current', 'current_variance',
                    'raw_current', 'electrometer_voltage', 'flags']

        if self.begin_time is None:
            raise RuntimeError('Attribute begin_time not present')

        order = np.argsort(self.begin_time)

        for attr in attrlist:
            seq = getattr(self, attr)
            sortedseq = [seq[i] for i in order]
            setattr(self, attr, sortedseq)

        self.field_data = {name: [seq[i] for i in order] for name, seq in self.field_data.items()}

    def split_electrometer_groups(self, vect: list[Any]) -> dict[str, list[Any]]:
        return {n: vect[b:e + 1] for n, (b, e) in self.electrometer_groups.items()}

    def get_record(self, i: int) -> dict[str, Any]:
        return {
            'begin_time_str': self.begin_time_str[i],
            'end_time_str': self.end_time_str[i],
            'opmode': self.opmode[i],
            'current': self.split_electrometer_groups(self.current[i]),
            'current_variance': self.split_electrometer_groups(self.current_variance[i]),
            'raw_current': self.split_electrometer_groups(self.raw_current[i]),
            'electrometer_voltage': self.split_electrometer_groups(self.electrometer_voltage[i]),
            'flags': [self.flag_defs[fi] for fi in self.flags[i]],
            'parameters': {n: v[i] for n, v in self.field_data.items()}
        }

    def blocks(self):
        if not self.opmode:
            return []
        blocklist = []
        curmode = self.opmode[0]
        curblock = []
        for i, op in enumerate(self.opmode):
            if op == curmode:
                curblock.append(i)
            else:
                blocklist.append((curmode, np.array(curblock)))
                curblock = [i]
                curmode = op
        return blocklist

    def write(self, output: IO[str]):
        output.write('# Spectops records\n')

        header = {
            'file type': 'records',
            'opmodes': list(self.opmode_set),
            'electrometer names': self.electrometer_names,
            'electrometer groups': self.electrometer_groups,
            'parameters': [self.field_definitions[k] for k in self.field_ids],
            'total electrometers': self.num_electrometers
        }

        for line in yaml.safe_dump(header, allow_unicode=True).split('\n'):
            output.write('# ')
            output.write(line)
            output.write('\n')

        columns = ['begin_time', 'end_time', 'opmode'] + \
                  ['cur_{}'.format(i) for i in range(self.num_electrometers)] + \
                  ['curvar_{}'.format(i) for i in range(self.num_electrometers)] + \
                  ['rawcur_{}'.format(i) for i in range(self.num_electrometers)] + \
                  ['volt_{}'.format(i) for i in range(self.num_electrometers)] + \
                  self.field_ids + \
                  ['flags']

        output.write('\t'.join(columns))
        output.write('\n')

        flagkeys = []

        field_data_list = [self.field_data[field_id] for field_id in self.field_ids]

        for f in self.flag_defs:
            digest = hashlib.sha256(f.encode('utf8')).hexdigest()

            for i in range(2, len(digest)):
                d = digest[:i]
                if d not in flagkeys:
                    output.write('# flag {}: {}\n'.format(d, f))
                    flagkeys.append(d)
                    break

        for i in range(len(self.begin_time_str)):
            output.write(self.begin_time_str[i])
            output.write('\t')
            output.write(self.end_time_str[i])
            output.write('\t')
            output.write(self.opmode[i])
            output.write('\t')

            for c in self.current[i]:
                output.write(str(c))
                output.write('\t')
            for c in self.current_variance[i]:
                output.write(str(c))
                output.write('\t')
            for c in self.raw_current[i]:
                output.write(str(c))
                output.write('\t')
            for c in self.electrometer_voltage[i]:
                output.write(str(c))
                output.write('\t')
            for pl in field_data_list:
                output.write(str(pl[i]))
                output.write('\t')

            for f in self.flags[i]:
                output.write('!')
                output.write(flagkeys[f])

            output.write('\n')

    def parse_times(self):
        self.begin_time = [parse_spectops_time(t) for t in self.begin_time_str]
        self.end_time = [parse_spectops_time(t) for t in self.end_time_str]

    def to_dataframe(self):
        all_field_data = {
            'begin_time': self.begin_time,
            'end_time': self.end_time,
            'opmode': pd.Series(self.opmode, dtype="category"),
        }

        if self.electrometer_groups:
            groups = self.electrometer_groups
        else:
            groups = {'': (0, self.num_electrometers)}

        for name, (first, last) in groups.items():
            if len(name) != 0:
                name += '_'
            for i in range(first, last + 1):
                all_field_data[f'{name}electrometer_current_{i - first}'] = [x[i] for x in self.current]
            for i in range(first, last + 1):
                all_field_data[f'{name}electrometer_raw_current_{i - first}'] = [x[i] for x in self.raw_current]
            for i in range(first, last + 1):
                all_field_data[f'{name}electrometer_current_variance_{i - first}'] = [x[i] for x in
                                                                                      self.current_variance]
            for i in range(first, last + 1):
                all_field_data[f'{name}electrometer_voltage_{i - first}'] = [x[i] for x in self.electrometer_voltage]

        all_field_data.update(self.field_data)
        return pd.DataFrame.from_dict(all_field_data)


class ParsingError(Exception):
    pass


def read_records_date_range(template: str, begin_date: datetime.date, end_date: datetime.date) -> RecordsFiles:
    """
    Read a range of files for a period of dates. Missing files are ignored.
    :param template: String format template for the files. E.g. "instrument/data/{date:%Y%m%d}-block.records'
    :param begin_date: Begin date as datetime.date
    :param end_date: End date as datetime.date
    :return: RecordsFiles object with the data loaded
    """
    records = RecordsFiles()
    date = begin_date

    day = datetime.timedelta(days=1)

    while date < end_date:
        try:
            with open(template.format(date=date)) as f:
                records.load(f)
        except FileNotFoundError as _:
            pass
        date += day

    records.sort()
    return records


def init_spectrum_data(definition: dict[str, Any]) -> SpectraData:
    xpoints = definition['xpoints']
    scale = definition['scale']
    if scale == 'radius':
        xpoints = [x * 2 for x in xpoints]
        scale_enum = SpectrumScale.DIAMETER
    elif scale == 'diameter':
        scale_enum = SpectrumScale.DIAMETER
    elif scale == 'mobility':
        scale_enum = SpectrumScale.MOBILITY
    else:
        raise ParsingError(f'Unknonw spectrum scale: "{scale}"')

    sd = SpectraData(
        name=definition['name'],
        inverter_name=definition['inverter_name'],
        scale=scale_enum,
        xpoints=xpoints,
        xunit=definition['xunit'],
        yunit=definition['yunit'],
        hash=definition['hash'],
        xsize=len(xpoints),
    )

    return sd
