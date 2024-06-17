#!/bin/bash

# Define an array of cron job schedules and commands
CRON_JOBS=(
    "/1 * * * * /Utils/AutoClrC.sh"
    "/2 * * * * Utils/CheckIfCUIIsRunning.sh"
)

# Create a temporary file to store the current crontab
TEMP_CRON=$(mktemp)

# Check if the cron job is already present
crontab -l > "$TEMP_CRON"
for CRON_JOB in "${CRON_JOBS[@]}"; do
    if ! grep -q "$CRON_JOB" "$TEMP_CRON"; then
        # Append the new cron job to the temporary crontab file
        echo "$CRON_JOB" >> "$TEMP_CRON"
        echo "Cron job added: $CRON_JOB"
    else
        echo "Cron job already exists: $CRON_JOB"
    fi
done

# Install the new crontab from the temporary file
crontab "$TEMP_CRON"

# Remove the temporary file
rm "$TEMP_CRON"
