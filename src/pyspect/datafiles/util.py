import datetime
import numpy


def float_or_missing(x):
    sx = x.strip()
    if sx:
        return float(sx)
    else:
        return numpy.nan


_tz_cache = {}


def get_fixed_tz(m):
    if m in _tz_cache:
        return _tz_cache[m]
    else:
        x = datetime.timezone(datetime.timedelta(minutes=m), "local")
        _tz_cache[m] = x
        return x


def parse_spectops_time(s):
    # print s[5:7]
    s = s.strip()
    if len(s) == 0:
        return None

    if len(s) == 26:
        tz = datetime.timezone.utc
    else:
        tz = get_fixed_tz(int(s[26:29]) * 60 + int(s[30:31]))

    t = datetime.datetime(
        year=int(s[:4]),
        month=int(s[5:7]),
        day=int(s[8:10]),
        hour=int(s[11:13]),
        minute=int(s[14:16]),
        second=int(s[17:19]),
        microsecond=int(s[20:26]),
        tzinfo=tz)

    return t


def parse_spectops_time_notz(s):
    # print s[5:7]
    if len(s) == 0:
        return None

    t = datetime.datetime(
        year=int(s[:4]),
        month=int(s[5:7]),
        day=int(s[8:10]),
        hour=int(s[11:13]),
        minute=int(s[14:16]),
        second=int(s[17:19]),
        microsecond=int(s[20:26]),
        tzinfo=datetime.timezone.utc)

    return t
