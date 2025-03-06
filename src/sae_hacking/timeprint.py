#!/usr/bin/env python3

import datetime


def timeprint(*args):
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

    # Print timestamp followed by the original content
    print(timestamp, *args)
