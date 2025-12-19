### **README.md**

````markdown
# Raspberry Pi Power Monitor – VoltWatch

An enhanced power restoration monitoring system for Raspberry Pi.  
Logs power outages, computes historical insights, and sends human-friendly email notifications with CSV history attached.

---

## Features

- Logs power restoration events with timestamp, uptime, and reboot type.
- Detects clean vs unexpected shutdowns.
- Calculates outage analytics using historical CSV data:
  - Average outage duration
  - Total downtime today
  - Rapid power cycles
  - Peak outage hours
  - Power stability score
- Sends a detailed email report with insights and recommendations.
- Fully self-contained in one folder: script, CSV, `.env`, and clean shutdown flag.
- Human-friendly notifications (hours/minutes, short blip, mostly stable, warnings).

---

## Requirements

- Python 3.8+  
- Packages:
  ```bash
  pip install python-dotenv
````

* Access to an SMTP server (credentials stored in `.env`)

---

## Setup

1. Clone the repository to your Raspberry Pi:

   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. Create a `.env` file in the same folder as the script:

   ```env
   EMAIL_PASSWORD=your_app_password
   ```

3. Make sure the script is executable:

   ```bash
   chmod +x power_monitor.py
   ```

4. Optional: Run at boot using `crontab -e`:

   ```cron
   @reboot /usr/bin/python3 /home/pi/VoltWatch/power_monitor.py
   ```

---

## Files

* `power_monitor.py` – Main monitoring script
* `.env` – Environment variables (not committed to git)
* `power_restore_history.csv` – Historical log of power events
* `.clean_shutdown` – Flag to detect clean shutdown
* `power_monitor.log` – Script runtime log

---

## Usage

Simply run the script manually for testing:

```bash
python3 power_monitor.py
```

On boot, it will automatically send an email with the latest event and insights based on historical data.

---

## Notes

* Designed to run on Raspberry Pi/Linux systems.
* Stores all files in a single folder for simplicity.
* Email and CSV attachment handled automatically.

---

````

---

### **.gitignore**

```gitignore
# Environment variables
.env

# CSV data
power_restore_history.csv

# Logs
power_monitor.log

# Clean shutdown flag
.clean_shutdown

# Python cache
__pycache__/
*.pyc
````

---

This setup ensures:

* Sensitive info in `.env` is never committed.
* CSV history and logs are kept locally but ignored by git.
* The project is easy to clone and set up on a Raspberry Pi.

---