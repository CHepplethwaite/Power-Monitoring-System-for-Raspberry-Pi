#!/usr/bin/env python3
"""
Enhanced Power Monitoring System for Raspberry Pi
Logs power restoration events with advanced analytics and insights.
"""

import os
import csv
import socket
import smtplib
import statistics
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from dotenv import load_dotenv
import logging
from collections import defaultdict

# =========================
# CONFIGURATION
# =========================

class Config:
    """Configuration settings"""
    SCRIPT_DIR = Path(__file__).parent.absolute()
    CSV_FILENAME = "power_restore_history.csv"
    ENV_FILENAME = ".env"
    FLAG_FILENAME = ".clean_shutdown"
    
    SMTP_SERVER = "mail.tumpetech.com"
    SMTP_PORT = 587
    EMAIL_SENDER = "no-reply@tumpetech.com"
    EMAIL_RECIPIENTS = ["clifford@tumpetech.com"]
    
    RAPID_CYCLE_THRESHOLD = 300  # 5 minutes in seconds

# =========================
# LOGGING SETUP
# =========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.SCRIPT_DIR / 'power_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# ANALYTICS ENGINE
# =========================

class PowerAnalytics:
    """Analyze power outage data and generate insights"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.events = self._load_events()
        self.today_str = datetime.now().strftime('%Y-%m-%d')
    
    def _load_events(self):
        """Load events from CSV file"""
        events = []
        if not self.csv_path.exists():
            logger.info("No existing CSV file found")
            return events
        
        try:
            with open(self.csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields
                    if 'Uptime_Seconds' in row and row['Uptime_Seconds']:
                        try:
                            row['Uptime_Seconds'] = float(row['Uptime_Seconds'])
                        except:
                            row['Uptime_Seconds'] = 0
                    events.append(row)
            logger.info(f"Loaded {len(events)} historical events")
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
        
        return events
    
    def calculate_average_outage(self):
        """Calculate average outage duration across all events"""
        if not self.events:
            return 0, "No historical data"
        
        outages = []
        for event in self.events:
            if 'Uptime_Seconds' in event and event['Uptime_Seconds']:
                outages.append(event['Uptime_Seconds'])
        
        if not outages:
            return 0, "No outage data"
        
        avg_seconds = statistics.mean(outages)
        return avg_seconds, self._format_duration(avg_seconds)
    
    def calculate_today_downtime(self):
        """Calculate total downtime for today"""
        total_seconds = 0
        
        for event in self.events:
            if event.get('Date') == self.today_str and 'Uptime_Seconds' in event:
                total_seconds += event['Uptime_Seconds']
        
        return total_seconds, self._format_duration(total_seconds)
    
    def count_rapid_cycles_today(self):
        """Count rapid power cycles today"""
        count = 0
        for event in self.events:
            if (event.get('Date') == self.today_str and 
                event.get('Rapid_Cycle', '').upper() == 'YES'):
                count += 1
        return count
    
    def detect_peak_outage_hours(self):
        """Find most frequent hours for outages"""
        hour_counts = defaultdict(int)
        
        for event in self.events:
            if 'Time' in event and event['Time']:
                try:
                    hour = int(event['Time'].split(':')[0])
                    hour_counts[hour] += 1
                except:
                    continue
        
        if not hour_counts:
            return []
        
        max_count = max(hour_counts.values())
        peak_hours = [hour for hour, count in hour_counts.items() if count == max_count]
        
        # Format hours nicely
        formatted_hours = []
        for hour in sorted(peak_hours):
            if hour == 0:
                formatted_hours.append("12 AM")
            elif hour < 12:
                formatted_hours.append(f"{hour} AM")
            elif hour == 12:
                formatted_hours.append("12 PM")
            else:
                formatted_hours.append(f"{hour-12} PM")
        
        return formatted_hours, max_count
    
    def calculate_stability_score(self):
        """Calculate power stability score (0-100)"""
        if len(self.events) < 2:
            return 100, "Insufficient data for scoring"
        
        # Base score starts at 100
        score = 100
        
        # Penalty for average outage duration
        avg_outage_seconds, _ = self.calculate_average_outage()
        if avg_outage_seconds > 3600:  # More than 1 hour average
            score -= 40
        elif avg_outage_seconds > 600:  # More than 10 minutes average
            score -= 20
        elif avg_outage_seconds > 60:   # More than 1 minute average
            score -= 10
        
        # Penalty for rapid cycles today
        rapid_today = self.count_rapid_cycles_today()
        score -= rapid_today * 15
        
        # Penalty for many outages today
        today_outages = len([e for e in self.events if e.get('Date') == self.today_str])
        if today_outages > 5:
            score -= 30
        elif today_outages > 3:
            score -= 20
        elif today_outages > 1:
            score -= 10
        
        # Ensure score stays within 0-100 range
        score = max(0, min(100, int(score)))
        
        # Determine stability level
        if score >= 90:
            level = "Excellent"
        elif score >= 75:
            level = "Good"
        elif score >= 60:
            level = "Fair"
        elif score >= 40:
            level = "Poor"
        else:
            level = "Critical"
        
        return score, level
    
    def get_insights_summary(self):
        """Generate human-friendly insights summary"""
        insights = []
        
        # Average outage insight
        avg_seconds, avg_formatted = self.calculate_average_outage()
        if avg_seconds > 0:
            if avg_seconds < 60:
                insights.append(f"• Average outage is a short blip ({avg_formatted})")
            elif avg_seconds < 300:
                insights.append(f"• Average outage is brief ({avg_formatted})")
            elif avg_seconds < 1800:
                insights.append(f"• Average outage is moderate ({avg_formatted})")
            else:
                insights.append(f"• Average outage is lengthy ({avg_formatted})")
        
        # Today's downtime insight
        today_seconds, today_formatted = self.calculate_today_downtime()
        if today_seconds > 0:
            if today_seconds < 300:
                insights.append(f"• Today's total downtime: {today_formatted} (minimal)")
            elif today_seconds < 1800:
                insights.append(f"• Today's total downtime: {today_formatted} (noticeable)")
            else:
                insights.append(f"• Today's total downtime: {today_formatted} (significant)")
        
        # Rapid cycles insight
        rapid_count = self.count_rapid_cycles_today()
        if rapid_count > 0:
            insights.append(f"• ⚠️ {rapid_count} rapid power cycle(s) today")
        
        # Peak hours insight
        peak_hours, peak_count = self.detect_peak_outage_hours()
        if peak_hours and peak_count > 1:
            if len(peak_hours) == 1:
                insights.append(f"• Peak outage time: {peak_hours[0]} ({peak_count} events)")
            else:
                insights.append(f"• Peak outage times: {', '.join(peak_hours)} ({peak_count} events each)")
        
        # Stability insight
        score, level = self.calculate_stability_score()
        insights.append(f"• Power stability: {level} ({score}/100)")
        
        return insights
    
    def _format_duration(self, seconds):
        """Format seconds into human-readable duration"""
        seconds = int(seconds)
        
        if seconds < 60:
            return f"{seconds} second{'s' if seconds != 1 else ''}"
        
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        
        hours = minutes // 60
        remaining_minutes = minutes % 60
        
        if remaining_minutes == 0:
            return f"{hours} hour{'s' if hours != 1 else ''}"
        else:
            return f"{hours}h {remaining_minutes}m"

# =========================
# CORE FUNCTIONS
# =========================

def load_environment():
    """Load email password from .env file"""
    env_path = Config.SCRIPT_DIR / Config.ENV_FILENAME
    if not env_path.exists():
        logger.error(f"Missing .env file at {env_path}")
        raise FileNotFoundError("Create a .env file with EMAIL_PASSWORD")
    
    load_dotenv(env_path)
    email_password = os.environ.get("EMAIL_PASSWORD")
    
    if not email_password:
        logger.error("EMAIL_PASSWORD not found in .env")
        raise ValueError("Add EMAIL_PASSWORD to .env file")
    
    return email_password

def get_system_info():
    """Collect system information"""
    info = {}
    
    try:
        info['hostname'] = socket.gethostname()
    except:
        info['hostname'] = "Unknown"
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        info['ip_address'] = s.getsockname()[0]
        s.close()
    except:
        info['ip_address'] = "Unknown"
    
    return info

def get_uptime_seconds():
    """Get system uptime from /proc/uptime"""
    try:
        with open('/proc/uptime', 'r') as f:
            return float(f.readline().split()[0])
    except Exception as e:
        logger.error(f"Error reading uptime: {e}")
        return 0

def detect_reboot_type():
    """Check if reboot was clean or unexpected"""
    flag_path = Config.SCRIPT_DIR / Config.FLAG_FILENAME
    
    if flag_path.exists():
        try:
            flag_path.unlink()
        except:
            pass
        return "Clean Reboot"
    return "Unexpected Power Loss"

def format_outage_duration(uptime_seconds, analytics):
    """Format outage duration with historical context"""
    if uptime_seconds < 60:
        desc = "brief blip"
    elif uptime_seconds < 300:
        desc = "short outage"
    elif uptime_seconds < 1800:
        desc = "moderate outage"
    elif uptime_seconds < 3600:
        desc = "long outage"
    else:
        desc = "extended outage"
    
    # Compare with historical average
    avg_seconds, _ = analytics.calculate_average_outage()
    if avg_seconds > 0:
        if uptime_seconds < avg_seconds * 0.5:
            desc += " (shorter than usual)"
        elif uptime_seconds > avg_seconds * 1.5:
            desc += " (longer than usual)"
    
    # Format for display
    if uptime_seconds < 60:
        formatted = f"{int(uptime_seconds)} seconds"
    else:
        minutes = int(uptime_seconds // 60)
        if minutes < 60:
            formatted = f"{minutes} minutes"
        else:
            hours = minutes // 60
            mins = minutes % 60
            formatted = f"{hours}h {mins}m"
    
    return formatted, desc

def log_event_to_csv(csv_path, event_data):
    """Append event to CSV file"""
    file_exists = csv_path.exists()
    
    try:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Date', 'Time', 'Event', 'Reboot_Type', 'IP_Address',
                'Outage_Duration', 'Rapid_Cycle', 'Uptime_Seconds'
            ])
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(event_data)
        
        logger.info(f"Event logged: {event_data['Date']} {event_data['Time']}")
        return True
    except Exception as e:
        logger.error(f"Error writing to CSV: {e}")
        return False

def create_email_content(system_info, event_data, analytics):
    """Generate comprehensive email content"""
    now = datetime.now()
    
    # Get analytics
    avg_seconds, avg_formatted = analytics.calculate_average_outage()
    today_seconds, today_formatted = analytics.calculate_today_downtime()
    rapid_count = analytics.count_rapid_cycles_today()
    score, stability_level = analytics.calculate_stability_score()
    peak_hours, peak_count = analytics.detect_peak_outage_hours()
    insights = analytics.get_insights_summary()
    
    # Determine emoji for subject
    if score >= 80:
        emoji = "✅"
    elif score >= 60:
        emoji = "⚠️"
    else:
        emoji = "🔴"
    
    # Format outage description
    outage_formatted, outage_desc = format_outage_duration(
        event_data['Uptime_Seconds'], analytics
    )
    
    # Create email content
    content = f"""
🔋 POWER RESTORATION REPORT
{'=' * 50}

📡 SYSTEM INFORMATION
• Device: {system_info['hostname']}
• IP Address: {system_info['ip_address']}
• Report Time: {now.strftime('%Y-%m-%d %H:%M:%S')}

⚡ CURRENT EVENT
• Status: Power Restored
• Reboot Type: {event_data['Reboot_Type']}
• Outage Duration: {outage_formatted} ({outage_desc})
• Rapid Cycle: {event_data['Rapid_Cycle']}

📊 POWER ANALYTICS
• Average Outage: {avg_formatted}
• Today's Total Downtime: {today_formatted}
• Rapid Cycles Today: {rapid_count}
• Power Stability: {stability_level} ({score}/100)
• Total Events Recorded: {len(analytics.events) + 1}

📈 KEY INSIGHTS
"""
    
    # Add insights
    for insight in insights:
        content += f"{insight}\n"
    
    # Add peak hours if available
    if peak_hours:
        if len(peak_hours) == 1:
            content += f"• Most outages occur around {peak_hours[0]}\n"
        else:
            content += f"• Outages are most frequent around {', '.join(peak_hours)}\n"
    
    # Add recommendations
    content += f"""
💡 RECOMMENDATIONS
"""
    
    if rapid_count > 2:
        content += "• ⚠️ Multiple rapid cycles detected - consider UPS installation\n"
    elif event_data['Rapid_Cycle'] == 'YES':
        content += "• ⚠️ Rapid cycle detected - monitor power quality\n"
    
    if score < 60:
        content += "• 🔧 Power stability is concerning - investigate power source\n"
    elif score < 80:
        content += "• 📝 Monitor power quality - stability could be improved\n"
    else:
        content += "• ✅ Power quality appears stable\n"
    
    if today_seconds > 3600:
        content += "• ⏰ Significant downtime today - check for scheduled outages\n"
    
    content += f"""
{'=' * 50}
📎 Full history attached as CSV
Generated by Raspberry Pi Power Monitor
"""
    
    return content.strip(), f"{emoji} ZESCO Power Restored - {system_info['hostname']}"

def send_email(email_password, subject, body, csv_path):
    """Send email with CSV attachment"""
    try:
        msg = EmailMessage()
        msg["From"] = Config.EMAIL_SENDER
        msg["To"] = ", ".join(Config.EMAIL_RECIPIENTS)
        msg["Subject"] = subject
        
        msg.set_content(body)
        
        # Attach CSV
        if csv_path.exists():
            with open(csv_path, 'rb') as f:
                csv_data = f.read()
                msg.add_attachment(
                    csv_data,
                    maintype='text',
                    subtype='csv',
                    filename='power_restore_history.csv'
                )
        
        # Send email
        with smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT) as server:
            server.starttls()
            server.login(Config.EMAIL_SENDER, email_password)
            server.send_message(msg)
        
        logger.info("Email sent successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

def create_clean_shutdown_flag():
    """Create flag for clean shutdown detection"""
    try:
        flag_path = Config.SCRIPT_DIR / Config.FLAG_FILENAME
        with open(flag_path, 'w') as f:
            f.write(f"Created: {datetime.now().isoformat()}\n")
        logger.info("Clean shutdown flag created")
        return True
    except Exception as e:
        logger.error(f"Error creating flag: {e}")
        return False

# =========================
# MAIN EXECUTION
# =========================

def main():
    """Main function"""
    logger.info("Starting enhanced power monitor...")
    
    try:
        # 1. Load configuration
        email_password = load_environment()
        
        # 2. Get system information
        system_info = get_system_info()
        logger.info(f"System: {system_info['hostname']} ({system_info['ip_address']})")
        
        # 3. Initialize analytics with historical data
        csv_path = Config.SCRIPT_DIR / Config.CSV_FILENAME
        analytics = PowerAnalytics(csv_path)
        
        # 4. Get current event data
        now = datetime.now()
        uptime_seconds = get_uptime_seconds()
        reboot_type = detect_reboot_type()
        
        # 5. Format outage information
        outage_formatted, outage_desc = format_outage_duration(uptime_seconds, analytics)
        
        # 6. Check for rapid cycle
        rapid_cycle = "YES" if uptime_seconds < Config.RAPID_CYCLE_THRESHOLD else "NO"
        
        # 7. Prepare event data for CSV
        event_data = {
            'Date': now.strftime('%Y-%m-%d'),
            'Time': now.strftime('%H:%M:%S'),
            'Event': 'Power Restored',
            'Reboot_Type': reboot_type,
            'IP_Address': system_info['ip_address'],
            'Outage_Duration': outage_desc,
            'Rapid_Cycle': rapid_cycle,
            'Uptime_Seconds': uptime_seconds
        }
        
        # 8. Log event to CSV
        if not log_event_to_csv(csv_path, event_data):
            logger.warning("Failed to log event, but continuing...")
        
        # 9. Generate and send email
        email_body, email_subject = create_email_content(system_info, event_data, analytics)
        
        if not send_email(email_password, email_subject, email_body, csv_path):
            logger.error("Failed to send email notification")
        
        # 10. Create clean shutdown flag for next boot
        create_clean_shutdown_flag()
        
        logger.info("Power monitor completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()