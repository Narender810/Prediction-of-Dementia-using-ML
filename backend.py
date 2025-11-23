import smtplib
import time

def send_email(name, email, message):
    smtp_server = "smtp.gmail.com"
    port = 587 
    receiver_email = "none@gmail.com"  
    password = "app_passwords"  

    with smtplib.SMTP(smtp_server, port) as server:
        server.starttls()
        server.login(email, password)

        subject = f"New message from {name}"
        body = f"From: {name}\nEmail: {email}\n\n{message}"

        server.sendmail(email, receiver_email, f"Subject: {subject}\n\n{body}")

def process_contact_form(name, email, message):
    if not name or not email or not message:
        return False, "Please fill out all fields."
    else:
        time.sleep(2)
        try:
            send_email(name, email, message)
            return True, "Email sent successfully!"
        except Exception as e:
            return False, f"An error occurred while sending the email: {str(e)}"
