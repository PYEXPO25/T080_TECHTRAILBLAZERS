import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sender_email = "vinukarthick6@gmail.com"
receiver_email = "mourishantonyc@gmail.com"
password = "VkmasteVKGP29"  # Use App Password for security

msg = MIMEMultipart()
msg["From"] = sender_email
msg["To"] = receiver_email
msg["Subject"] = "🚨 Smart Policing Alert!"
msg.attach(MIMEText("🚨 Alert! Suspicious activity detected.", "plain"))

server = smtplib.SMTP("smtp.gmail.com", 587)
server.starttls()
server.login(sender_email, password)
server.sendmail(sender_email, receiver_email, msg.as_string())
server.quit()

print("✅ Email sent successfully!")
