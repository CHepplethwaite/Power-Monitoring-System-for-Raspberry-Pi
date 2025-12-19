#!/usr/bin/env python3
"""
Production-Ready Power Monitoring System for Raspberry Pi
Enhanced with SQLite, reliability features, and production optimizations
"""

import os
import sys
import sqlite3
import socket
import smtplib
import statistics
import json
import time
import signal
import atexit
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict
import logging
import logging.handlers
from dataclasses import dataclass
import hashlib
import fcntl
import threading
from contextlib import contextmanager
from functools import wraps
import pickle
import csv
import io
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# =========================
# CONFIGURATION CLASS
# =========================

@dataclass
class Config:
    """Application configuration with defaults"""
    # Paths
    SCRIPT_DIR: Path = Path(__file__).parent.absolute()
    DATA_DIR: Path = SCRIPT_DIR / "data"
    LOG_DIR: Path = SCRIPT_DIR / "logs"
    CONFIG_FILE: Path = SCRIPT_DIR / "config.json"
    
    # Database
    DB_FILENAME: str = "power_monitor.db"
    DB_RETENTION_DAYS: int = 90  # Automatically archive data older than this
    DB_BACKUP_DIR: Path = DATA_DIR / "backups"
    
    # Email
    SMTP_SERVER = os.getenv("SMTP_SERVER")
    SMTP_PORT = int(os.getenv("SMTP_PORT"))
    EMAIL_SENDER = os.getenv("EMAIL_SENDER")
    EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS").split(",")
    
    # Monitoring
    RAPID_CYCLE_THRESHOLD: int = 300  # 5 minutes in seconds
    CHECK_INTERVAL: int = 60  # Seconds between checks when running as service
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # Security
    ENV_PASSWORD_KEY: str = "EMAIL_PASSWORD"
    
    # Lock file for preventing overlapping runs
    LOCK_FILE: Path = SCRIPT_DIR / ".power_monitor.lock"
    
    def __post_init__(self):
        if self.EMAIL_RECIPIENTS is None:
            self.EMAIL_RECIPIENTS = ["clifford@tumpetech.com"]
        
        # Create directories
        self.DATA_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)
        self.DB_BACKUP_DIR.mkdir(exist_ok=True)

# =========================
# DATABASE MANAGER
# =========================

class DatabaseManager:
    """Manages SQLite database operations with atomic writes and cleanup"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS power_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    event TEXT NOT NULL,
                    reboot_type TEXT NOT NULL,
                    ip_address TEXT,
                    outage_duration TEXT,
                    rapid_cycle TEXT,
                    uptime_seconds REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_key TEXT NOT NULL,
                    metric_value TEXT NOT NULL,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(metric_key)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_power_events_date 
                ON power_events(date)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_power_events_created 
                ON power_events(created_at)
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30,  # 30 second timeout
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def insert_event(self, event_data: Dict[str, Any]) -> bool:
        """Insert event with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO power_events 
                        (date, time, event, reboot_type, ip_address, 
                         outage_duration, rapid_cycle, uptime_seconds, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event_data['date'],
                        event_data['time'],
                        event_data['event'],
                        event_data['reboot_type'],
                        event_data.get('ip_address'),
                        event_data.get('outage_duration'),
                        event_data.get('rapid_cycle'),
                        event_data.get('uptime_seconds'),
                        json.dumps(event_data.get('metadata', {}))
                    ))
                    conn.commit()
                return True
            except sqlite3.Error as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        return False
    
    def get_recent_events(self, days: int = 30) -> List[Dict]:
        """Get recent events with optional archival"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT date, time, event, reboot_type, ip_address,
                       outage_duration, rapid_cycle, uptime_seconds, metadata
                FROM power_events
                WHERE date >= date('now', ?)
                ORDER BY date DESC, time DESC
            """, (f'-{days} days',))
            
            events = []
            for row in cursor.fetchall():
                events.append(dict(row))
            return events
    
    def get_all_events(self) -> List[Dict]:
        """Get all events from database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT date, time, event, reboot_type, ip_address,
                       outage_duration, rapid_cycle, uptime_seconds, metadata
                FROM power_events
                ORDER BY date DESC, time DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_data(self, retention_days: int) -> int:
        """Archive and remove old data, return number of rows removed"""
        try:
            # First, export old data to CSV
            archive_date = (datetime.now() - timedelta(days=retention_days)).strftime('%Y-%m-%d')
            archive_file = Config.DB_BACKUP_DIR / f"archive_{archive_date}.csv"
            
            with self._get_connection() as conn:
                # Export old data
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM power_events 
                    WHERE date < date('now', ?)
                """, (f'-{retention_days} days',))
                
                old_data = cursor.fetchall()
                
                if old_data:
                    # Write to CSV
                    with open(archive_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([desc[0] for desc in cursor.description])
                        writer.writerows(old_data)
                    
                    # Delete old data
                    cursor.execute("""
                        DELETE FROM power_events 
                        WHERE date < date('now', ?)
                    """, (f'-{retention_days} days',))
                    
                    conn.commit()
                    return cursor.rowcount
            
            return 0
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0
    
    def backup_database(self) -> bool:
        """Create a backup of the database"""
        try:
            backup_file = Config.DB_BACKUP_DIR / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            with self._get_connection() as src_conn:
                with sqlite3.connect(backup_file) as dst_conn:
                    src_conn.backup(dst_conn)
            logger.info(f"Database backed up to {backup_file}")
            return True
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            return False

# =========================
# SECRETS MANAGER
# =========================

class SecretsManager:
    """Manages secure storage and retrieval of secrets"""
    
    @staticmethod
    def get_email_password() -> Optional[str]:
        """Get email password from environment variable"""
        # First try environment variable
        password = os.environ.get(Config.ENV_PASSWORD_KEY)
        
        if not password:
            # Fallback to .env file
            env_file = Config.SCRIPT_DIR / ".env"
            if env_file.exists():
                try:
                    with open(env_file, 'r') as f:
                        for line in f:
                            if line.startswith(f"{Config.ENV_PASSWORD_KEY}="):
                                password = line.split('=', 1)[1].strip()
                                break
                except Exception:
                    pass
        
        if not password:
            logger.error(f"Email password not found. Set {Config.ENV_PASSWORD_KEY} environment variable.")
        
        return password
    
    @staticmethod
    def mask_sensitive(text: str) -> str:
        """Mask sensitive information in logs"""
        if not text:
            return text
        
        # Simple masking for email addresses
        if '@' in text:
            parts = text.split('@')
            if len(parts[0]) > 2:
                return f"{parts[0][:2]}***@{parts[1]}"
        
        return text

# =========================
# RETRY DECORATOR
# =========================

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after error: {e}. Waiting {delay}s..."
                        )
                        time.sleep(delay)
            
            logger.error(f"All retries failed for {func.__name__}: {last_exception}")
            raise last_exception
        return wrapper
    return decorator

# =========================
# FILE LOCK MANAGER
# =========================

class FileLock:
    """Prevents overlapping script execution using file locks"""
    
    def __init__(self, lock_file: Path):
        self.lock_file = lock_file
        self.lock_fd = None
    
    def acquire(self, timeout: int = 10) -> bool:
        """Acquire lock with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                self.lock_fd = open(self.lock_file, 'w')
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Write PID to lock file
                self.lock_fd.write(str(os.getpid()))
                self.lock_fd.flush()
                
                logger.debug(f"Lock acquired: {self.lock_file}")
                return True
            except (IOError, BlockingIOError):
                if self.lock_fd:
                    self.lock_fd.close()
                    self.lock_fd = None
                
                time.sleep(0.1)
        
        logger.warning(f"Could not acquire lock after {timeout} seconds")
        return False
    
    def release(self):
        """Release lock"""
        if self.lock_fd:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            self.lock_fd.close()
            self.lock_fd = None
            
            # Clean up lock file
            try:
                self.lock_file.unlink()
            except:
                pass
            
            logger.debug("Lock released")

# =========================
# LOGGING SETUP
# =========================

def setup_logging(config: Config) -> logging.Logger:
    """Setup structured logging with rotation"""
    logger = logging.getLogger('PowerMonitor')
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler with rotation
    log_file = config.LOG_DIR / 'power_monitor.log'
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - '
        '%(filename)s:%(lineno)d - %(message)s'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
config = Config()
logger = setup_logging(config)

# =========================
# ANALYTICS ENGINE
# =========================

class PowerAnalytics:
    """Analyze power outage data and generate insights"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.events = self._load_events()
        self.today_str = datetime.now().strftime('%Y-%m-%d')
    
    def _load_events(self) -> List[Dict]:
        """Load events from database with parsed metadata"""
        try:
            events = self.db_manager.get_recent_events(days=30)
            # Parse metadata to get actual outage seconds
            parsed_events = []
            for event in events:
                parsed_event = dict(event)
                metadata = event.get('metadata')
                if metadata:
                    try:
                        metadata_dict = json.loads(metadata)
                        parsed_event['actual_outage_seconds'] = metadata_dict.get('actual_outage_seconds', 0.0)
                    except (json.JSONDecodeError, TypeError):
                        parsed_event['actual_outage_seconds'] = 0.0
                else:
                    parsed_event['actual_outage_seconds'] = 0.0
                parsed_events.append(parsed_event)
            return parsed_events
        except Exception as e:
            logger.error(f"Error loading events: {e}")
            return []
    
    def calculate_average_outage(self) -> Tuple[float, str]:
        """Calculate average outage duration using actual outage seconds"""
        if not self.events:
            return 0.0, "No historical data"
        
        outages = []
        for event in self.events:
            # Use actual_outage_seconds from parsed metadata
            outage_seconds = event.get('actual_outage_seconds', 0.0)
            if outage_seconds and float(outage_seconds) > 0:
                outages.append(float(outage_seconds))
        
        if not outages:
            return 0.0, "No outage data"
        
        avg_seconds = statistics.mean(outages)
        return avg_seconds, self._format_duration(avg_seconds)
    
    def calculate_today_downtime(self) -> Tuple[float, str]:
        """Calculate total downtime for today using actual outage seconds"""
        total_seconds = 0.0
        
        for event in self.events:
            if event.get('date') == self.today_str:
                # Use actual_outage_seconds from parsed metadata
                outage_seconds = event.get('actual_outage_seconds', 0.0)
                if outage_seconds:
                    total_seconds += float(outage_seconds)
        
        return total_seconds, self._format_duration(total_seconds)
    
    def count_rapid_cycles_today(self) -> int:
        """Count rapid power cycles today"""
        count = 0
        for event in self.events:
            if (event.get('date') == self.today_str and 
                event.get('rapid_cycle', '').upper() == 'YES'):
                count += 1
        return count
    
    def detect_peak_outage_hours(self) -> Tuple[List[str], int]:
        """Find most frequent hours for outages"""
        hour_counts = defaultdict(int)
        
        for event in self.events:
            event_time = event.get('time')
            if event_time:
                try:
                    hour = int(event_time.split(':')[0])
                    hour_counts[hour] += 1
                except (ValueError, IndexError):
                    continue
        
        if not hour_counts:
            return [], 0
        
        max_count = max(hour_counts.values())
        peak_hours = [hour for hour, count in hour_counts.items() if count == max_count]
        
        # Format hours
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
    
    def calculate_stability_score(self) -> Tuple[int, str]:
        """Calculate power stability score (0-100)"""
        if len(self.events) < 2:
            return 100, "Insufficient data"
        
        score = 100
        
        # Penalty for average outage duration
        avg_outage_seconds, _ = self.calculate_average_outage()
        if avg_outage_seconds > 3600:
            score -= 40
        elif avg_outage_seconds > 600:
            score -= 20
        elif avg_outage_seconds > 60:
            score -= 10
        
        # Penalty for rapid cycles
        rapid_today = self.count_rapid_cycles_today()
        score -= rapid_today * 15
        
        # Penalty for many outages today
        today_outages = len([e for e in self.events if e.get('date') == self.today_str])
        if today_outages > 5:
            score -= 30
        elif today_outages > 3:
            score -= 20
        elif today_outages > 1:
            score -= 10
        
        # Ensure score is within bounds
        score = max(0, min(100, int(score)))
        
        # Determine level
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
    
    def get_insights_summary(self) -> List[str]:
        """Generate human-friendly insights"""
        insights = []
        
        # Average outage
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
        
        # Today's downtime
        today_seconds, today_formatted = self.calculate_today_downtime()
        if today_seconds > 0:
            if today_seconds < 300:
                insights.append(f"• Today's total downtime: {today_formatted} (minimal)")
            elif today_seconds < 1800:
                insights.append(f"• Today's total downtime: {today_formatted} (noticeable)")
            else:
                insights.append(f"• Today's total downtime: {today_formatted} (significant)")
        
        # Rapid cycles
        rapid_count = self.count_rapid_cycles_today()
        if rapid_count > 0:
            insights.append(f"• ⚠️ {rapid_count} rapid power cycle(s) today")
        
        # Peak hours
        peak_hours, peak_count = self.detect_peak_outage_hours()
        if peak_hours and peak_count > 1:
            if len(peak_hours) == 1:
                insights.append(f"• Peak outage time: {peak_hours[0]} ({peak_count} events)")
            else:
                insights.append(f"• Peak outage times: {', '.join(peak_hours)} ({peak_count} events each)")
        
        # Stability
        score, level = self.calculate_stability_score()
        insights.append(f"• Power stability: {level} ({score}/100)")
        
        return insights
    
    def _format_duration(self, seconds: float) -> str:
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
# EMAIL SERVICE
# =========================

class EmailService:
    """Handles email sending with retry logic and HTML formatting"""
    
    def __init__(self, config: Config):
        self.config = config
    
    @retry_with_backoff(max_retries=3)
    def send_power_restoration_email(
        self,
        system_info: Dict,
        event_data: Dict,
        analytics: PowerAnalytics,
        db_manager: DatabaseManager
    ) -> bool:
        """Send comprehensive power restoration email"""
        try:
            email_password = SecretsManager.get_email_password()
            if not email_password:
                return False
            
            # Generate email content
            html_content, text_content, subject = self._create_email_content(
                system_info, event_data, analytics
            )
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.config.EMAIL_SENDER
            msg['To'] = ', '.join(self.config.EMAIL_RECIPIENTS)
            msg['Subject'] = subject
            
            # Attach text and HTML versions
            msg.attach(MIMEText(text_content, 'plain'))
            msg.attach(MIMEText(html_content, 'html'))
            
            # Attach CSV export
            csv_attachment = self._create_csv_attachment(db_manager)
            if csv_attachment:
                msg.attach(csv_attachment)
            
            # Send email
            with smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT) as server:
                server.starttls()
                server.login(self.config.EMAIL_SENDER, email_password)
                server.send_message(msg)
            
            logger.info("Email sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def _create_email_content(
        self,
        system_info: Dict,
        event_data: Dict,
        analytics: PowerAnalytics
    ) -> Tuple[str, str, str]:
        """Generate HTML and text email content"""
        now = datetime.now()
        
        # Get analytics
        avg_seconds, avg_formatted = analytics.calculate_average_outage()
        today_seconds, today_formatted = analytics.calculate_today_downtime()
        rapid_count = analytics.count_rapid_cycles_today()
        score, stability_level = analytics.calculate_stability_score()
        peak_hours, peak_count = analytics.detect_peak_outage_hours()
        insights = analytics.get_insights_summary()
        
        # Determine emoji
        if score >= 80:
            emoji = "✅"
        elif score >= 60:
            emoji = "⚠️"
        else:
            emoji = "🔴"
        
        # HTML Content
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 30px; border-radius: 10px 10px 0 0; }}
        .section {{ background: #f9f9f9; padding: 20px; margin: 10px 0; border-radius: 5px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                       gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: white; padding: 15px; border-radius: 5px; 
                       box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
        .insight-list {{ list-style: none; padding: 0; }}
        .insight-list li {{ padding: 8px 0; border-bottom: 1px solid #eee; }}
        .status-excellent {{ color: #28a745; }}
        .status-good {{ color: #17a2b8; }}
        .status-fair {{ color: #ffc107; }}
        .status-poor {{ color: #dc3545; }}
        .status-critical {{ color: #dc3545; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .recommendation {{ background: #fff3cd; border-left: 4px solid #ffc107; 
                         padding: 15px; margin: 15px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔋 Power Restoration Report</h1>
            <p>Generated on {now.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📡 System Information</h2>
            <table>
                <tr><th>Device</th><td>{system_info['hostname']}</td></tr>
                <tr><th>IP Address</th><td>{system_info['ip_address']}</td></tr>
                <tr><th>Report Time</th><td>{now.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>⚡ Current Event</h2>
            <table>
                <tr><th>Status</th><td>Power Restored</td></tr>
                <tr><th>Reboot Type</th><td>{event_data['reboot_type']}</td></tr>
                <tr><th>Outage Duration</th><td>{event_data['outage_duration']}</td></tr>
                <tr><th>Rapid Cycle</th><td>{event_data['rapid_cycle']}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>📊 Power Analytics</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div>Average Outage</div>
                    <div class="metric-value">{avg_formatted}</div>
                </div>
                <div class="metric-card">
                    <div>Today's Downtime</div>
                    <div class="metric-value">{today_formatted}</div>
                </div>
                <div class="metric-card">
                    <div>Rapid Cycles Today</div>
                    <div class="metric-value">{rapid_count}</div>
                </div>
                <div class="metric-card">
                    <div>Stability Score</div>
                    <div class="metric-value {f'status-{stability_level.lower()}'}">{score}/100</div>
                    <div>{stability_level}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Key Insights</h2>
            <ul class="insight-list">
"""
        
        for insight in insights:
            html_content += f"<li>{insight}</li>\n"
        
        html_content += """
            </ul>
        </div>
        
        <div class="section">
            <h2>💡 Recommendations</h2>
            <div class="recommendation">
"""
        
        # Add recommendations
        if rapid_count > 2:
            html_content += "<p>⚠️ Multiple rapid cycles detected - consider UPS installation</p>"
        elif event_data['rapid_cycle'] == 'YES':
            html_content += "<p>⚠️ Rapid cycle detected - monitor power quality</p>"
        
        if score < 60:
            html_content += "<p>🔧 Power stability is concerning - investigate power source</p>"
        elif score < 80:
            html_content += "<p>📝 Monitor power quality - stability could be improved</p>"
        else:
            html_content += "<p>✅ Power quality appears stable</p>"
        
        if today_seconds > 3600:
            html_content += "<p>⏰ Significant downtime today - check for scheduled outages</p>"
        
        html_content += """
            </div>
        </div>
        
        <div class="section">
            <p><em>📎 Full history attached as CSV</em></p>
            <p><em>Generated by Raspberry Pi Power Monitor</em></p>
        </div>
    </div>
</body>
</html>
"""
        
        # Text Content (fallback)
        text_content = f"""
POWER RESTORATION REPORT
{'=' * 50}

SYSTEM INFORMATION
• Device: {system_info['hostname']}
• IP Address: {system_info['ip_address']}
• Report Time: {now.strftime('%Y-%m-%d %H:%M:%S')}

CURRENT EVENT
• Status: Power Restored
• Reboot Type: {event_data['reboot_type']}
• Outage Duration: {event_data['outage_duration']}
• Rapid Cycle: {event_data['rapid_cycle']}

POWER ANALYTICS
• Average Outage: {avg_formatted}
• Today's Total Downtime: {today_formatted}
• Rapid Cycles Today: {rapid_count}
• Power Stability: {stability_level} ({score}/100)
• Total Events Recorded: {len(analytics.events) + 1}

KEY INSIGHTS
"""
        
        for insight in insights:
            text_content += f"{insight}\n"
        
        text_content += f"""
{'=' * 50}
Generated by Raspberry Pi Power Monitor
"""
        
        subject = f"{emoji} ZESCO Power Restored - {system_info['hostname']}"
        
        return html_content.strip(), text_content.strip(), subject
    
    def _create_csv_attachment(self, db_manager: DatabaseManager) -> Optional[MIMEBase]:
        """Create CSV attachment from database"""
        try:
            events = db_manager.get_all_events()
            if not events:
                return None
            
            # Create CSV in memory
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=[
                'Date', 'Time', 'Event', 'Reboot_Type', 'IP_Address',
                'Outage_Duration', 'Rapid_Cycle', 'Uptime_Seconds'
            ])
            
            writer.writeheader()
            for event in events:
                writer.writerow({
                    'Date': event['date'],
                    'Time': event['time'],
                    'Event': event['event'],
                    'Reboot_Type': event['reboot_type'],
                    'IP_Address': event['ip_address'],
                    'Outage_Duration': event['outage_duration'],
                    'Rapid_Cycle': event['rapid_cycle'],
                    'Uptime_Seconds': event['uptime_seconds']
                })
            
            # Create attachment
            attachment = MIMEBase('text', 'csv')
            attachment.set_payload(output.getvalue())
            encoders.encode_base64(attachment)
            attachment.add_header(
                'Content-Disposition',
                'attachment',
                filename=f'power_history_{datetime.now().strftime("%Y%m%d")}.csv'
            )
            
            return attachment
            
        except Exception as e:
            logger.error(f"Error creating CSV attachment: {e}")
            return None

# =========================
# CORE SYSTEM FUNCTIONS
# =========================

def get_system_info() -> Dict[str, str]:
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

def get_uptime_seconds() -> float:
    """Get system uptime from /proc/uptime"""
    try:
        with open('/proc/uptime', 'r') as f:
            return float(f.readline().split()[0])
    except Exception as e:
        logger.error(f"Error reading uptime: {e}")
        return 0.0

def detect_reboot_type() -> str:
    """Check if reboot was clean or unexpected"""
    flag_path = config.SCRIPT_DIR / ".clean_shutdown"
    
    if flag_path.exists():
        try:
            flag_path.unlink()
        except:
            pass
        return "Clean Reboot"
    return "Unexpected Power Loss"

def create_clean_shutdown_flag() -> bool:
    """Create flag for clean shutdown detection"""
    try:
        flag_path = config.SCRIPT_DIR / ".clean_shutdown"
        with open(flag_path, 'w') as f:
            f.write(f"Created: {datetime.now().isoformat()}\n")
        logger.info("Clean shutdown flag created")
        return True
    except Exception as e:
        logger.error(f"Error creating flag: {e}")
        return False

# =========================
# MAIN APPLICATION
# =========================

class PowerMonitor:
    """Main application class"""
    
    def __init__(self):
        self.config = Config()
        self.file_lock = FileLock(self.config.LOCK_FILE)
        self.db_manager = DatabaseManager(self.config.DATA_DIR / self.config.DB_FILENAME)
        self.email_service = EmailService(self.config)
        self.is_service_mode = False

    def get_last_power_event(self) -> Optional[Dict]:
        """Get the most recent power event from database"""
        try:
            with self.db_manager._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT date, time, event, uptime_seconds, metadata
                    FROM power_events 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                row = cursor.fetchone()
                if row:
                    event_dict = dict(row)
                    # Parse metadata if exists
                    metadata = event_dict.get('metadata')
                    if metadata:
                        try:
                            metadata_dict = json.loads(metadata)
                            event_dict['actual_outage_seconds'] = metadata_dict.get('actual_outage_seconds', 0.0)
                        except (json.JSONDecodeError, TypeError):
                            event_dict['actual_outage_seconds'] = 0.0
                    return event_dict
        except Exception as e:
            logger.error(f"Error getting last power event: {e}")
        return None
        
    def run_once(self) -> bool:
        """Run single monitoring cycle"""
        if not self.file_lock.acquire():
            logger.warning("Another instance is already running")
            return False
        
        try:
            logger.info("Starting power monitor cycle...")
            
            # 1. Get system information
            system_info = get_system_info()
            logger.info(f"System: {system_info['hostname']} ({system_info['ip_address']})")
            
            
            # 2. Get current event data
            now = datetime.now()
            uptime_seconds = get_uptime_seconds()  # This is time since last boot
            reboot_type = detect_reboot_type()

            # 3. Calculate actual outage duration by comparing with last event
            last_event = self.get_last_power_event()
            outage_seconds = 0.0

            if last_event:
                # Calculate time difference between now and last event
                last_event_time = datetime.strptime(
                    f"{last_event['date']} {last_event['time']}", 
                    "%Y-%m-%d %H:%M:%S"
                )
                time_since_last_event = (now - last_event_time).total_seconds()
                
                # Outage duration is time since last event minus current uptime
                # This assumes system was down between last event and boot time
                outage_seconds = max(0.0, time_since_last_event - uptime_seconds)
                
                # Check for rapid cycle based on outage duration (not uptime)
                if outage_seconds < self.config.RAPID_CYCLE_THRESHOLD:
                    rapid_cycle = "YES"
                else:
                    rapid_cycle = "NO"
            else:
                # First run or no previous events
                rapid_cycle = "NO"
                outage_seconds = 0.0  # Can't determine outage duration on first run

            # 4. Format outage information for display
            if outage_seconds < 60:
                outage_desc = "brief blip"
                formatted = f"{int(outage_seconds)} seconds"
            elif outage_seconds < 300:
                outage_desc = "short outage"
                formatted = f"{int(outage_seconds // 60)} minutes"
            elif outage_seconds < 1800:
                outage_desc = "moderate outage"
                minutes = int(outage_seconds // 60)
                formatted = f"{minutes} minute{'s' if minutes != 1 else ''}"
            else:
                outage_desc = "long outage"
                hours = int(outage_seconds // 3600)
                minutes = int((outage_seconds % 3600) // 60)
                if minutes == 0:
                    formatted = f"{hours} hour{'s' if hours != 1 else ''}"
                else:
                    formatted = f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"

            # Also format uptime for metadata
            if uptime_seconds < 60:
                uptime_formatted = f"{int(uptime_seconds)} seconds"
            elif uptime_seconds < 3600:
                uptime_formatted = f"{int(uptime_seconds // 60)} minutes"
            else:
                hours = int(uptime_seconds // 3600)
                minutes = int((uptime_seconds % 3600) // 60)
                uptime_formatted = f"{hours}h {minutes}m"

            event_data = {
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "event": "Power Restored",
                "reboot_type": reboot_type,
                "ip_address": system_info.get("ip_address"),
                "outage_duration": formatted,  # Now correctly shows actual outage duration
                "rapid_cycle": rapid_cycle,
                "uptime_seconds": uptime_seconds,  # Time since last boot
                "metadata": {
                    "outage_desc": outage_desc,
                    "uptime_formatted": uptime_formatted,
                    "actual_outage_seconds": outage_seconds
                }
            }

            # 5. Insert into database
            self.db_manager.insert_event(event_data)

            # 6. Analytics
            analytics = PowerAnalytics(self.db_manager)

            # 7. Send email notification
            self.email_service.send_power_restoration_email(
                system_info, event_data, analytics, self.db_manager
            )

            # 8. Cleanup old data
            removed = self.db_manager.cleanup_old_data(self.config.DB_RETENTION_DAYS)
            if removed:
                logger.info(f"Cleaned up {removed} old records")

            logger.info("Power monitor cycle completed successfully")
            return True

        finally:
            self.file_lock.release()

    def run_service(self):
        """Run in continuous service mode"""
        self.is_service_mode = True
        logger.info("Starting Power Monitor in service mode...")
        try:
            while True:
                self.run_once()
                time.sleep(self.config.CHECK_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Service interrupted by user")
        except Exception as e:
            logger.error(f"Service encountered an error: {e}")
        finally:
            logger.info("Power Monitor service stopped")

# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    monitor = PowerMonitor()

    # Optionally create clean shutdown flag on normal exit
    def handle_exit(signum, frame):
        logger.info("Exiting... Creating clean shutdown flag")
        create_clean_shutdown_flag()
        sys.exit(0)

    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, handle_exit)

    # Run once or as a service based on command-line args
    if len(sys.argv) > 1 and sys.argv[1].lower() == "service":
        monitor.run_service()
    else:
        monitor.run_once()