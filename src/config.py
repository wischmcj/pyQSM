from __future__ import annotations

import logging
import logging.config
import os

import toml
import yaml

cwd = os.getcwd()
print(f"Current working directory: {cwd}")
# Read in environment variables, set defaults if not present
package_location = os.path.dirname(__file__)


config_file = os.environ.get("PY_QSM_CONFIG", f"{package_location}/canopyhydro_config.toml")
log_config = os.environ.get("PY_QSM_LOG_CONFIG", f"{package_location}/log.yml")

try:
    with open(log_config) as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
except Exception as error:
    print(f"Error loading log config {error}")

log = logging.getLogger('main')
