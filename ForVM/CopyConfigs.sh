#!/bin/bash
(crontab -l 2>/dev/null; echo "1 * * * * /usr/local/bin/clear_cache.sh") | crontab -
