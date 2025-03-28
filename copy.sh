#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
scp "$1" pivotal:
base=$(basename "$1")
ssh pivotal "sudo mv $base /var/www/html/"
echo http://3.10.22.91/"$base"
