import cv2 
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
from datetime import timedelta
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import stat

# Defining Flask App
app = Flask(__name__)

nimgs = 30

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


## A function to get names and roll numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


## A function to delete a user folder 
import os
import stat

import shutil
import os

def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        file_path = os.path.join(duser, i)
        # Ensure the file is writable
        os.chmod(file_path, stat.S_IWUSR)
        os.remove(file_path)
    os.rmdir(duser)

def calculate_total_sessions():
    # Example: Counting CSV files as sessions
    session_files = [f for f in os.listdir('Attendance') if f.endswith('.csv')]
    return len(session_files)

def calculate_attendance_percentage():
    attendance_percentages = {}
    total_sessions = calculate_total_sessions()  # Assuming you have a function to calculate total sessions
    
    for user_dir in os.listdir('static/faces'):
        username, user_id = user_dir.split('_')
        attendance_count = 0
        
        for attendance_file in os.listdir('Attendance'):
            if attendance_file.endswith('.csv'):
                df = pd.read_csv(os.path.join('Attendance', attendance_file))
                if df[(df['Name'] == username) & (df['Roll'] == int(user_id))].any(axis=None):
                    attendance_count += 1
        
        attendance_percentages[user_dir] = (attendance_count / total_sessions) * 100 if total_sessions > 0 else 0
    
    return attendance_percentages



################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    if 'user' in session:
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)
    return redirect(url_for('login'))


@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    attendance_percentages = calculate_attendance_percentage()  # Calculate attendance percentages

    # Combine names, rolls into a list of tuples and include attendance percentages
    users_info_with_attendance = []
    for name, roll in zip(names, rolls):
        key = f"{name}_{roll}"
        attendance_percentage = attendance_percentages.get(key, 0)  # Default to 0 if not found
        users_info_with_attendance.append((name, roll, attendance_percentage))

   # Sort users_info_with_attendance by the Roll ID in ascending order
    users_info_with_attendance_sorted = sorted(users_info_with_attendance, key=lambda x: x[1])

    # Pass the sorted list to the template
    return render_template('listusers.html', users_info=users_info_with_attendance_sorted, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## Delete functionality
from flask import redirect, url_for

@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    name = request.args.get('name')
    roll = request.args.get('roll')
    folder_name = f"{name}_{roll}"
    deletefolder(f'static/faces/{folder_name}')

    # Format today's date
    datetoday = datetime.now().strftime('%Y-%m-%d')
    attendance_file_path = f'Attendance/Attendance-{datetoday}.csv'
    
    # Check if today's attendance file exists
    if os.path.exists(attendance_file_path):
        # Load the attendance data
        df = pd.read_csv(attendance_file_path)
        
        # Filter the DataFrame to exclude the user being deleted
        updated_df = df[(df['Name'] != name) | (df['Roll'] != roll)]
        
        # Save the updated DataFrame
        updated_df.to_csv(attendance_file_path, index=False)

    ## if all the faces are deleted, delete the trained file...
    if not os.listdir(os.path.join('static', 'faces')):
        os.remove(os.path.join('static', 'face_recognition_model.pkl'))
    
    try:
        train_model()
    except Exception as e:
        print(f"Error training model: {e}")
    
    return redirect(url_for('listusers'))


# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Attendance', cv2.WINDOW_NORMAL)  # Create a window that can be resized
    cv2.setWindowProperty('Attendance', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Make the window fullscreen
    cv2.setWindowProperty('Attendance', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)  # Set it back to normal but keeps the window resizable
    cv2.setWindowProperty('Attendance', cv2.WND_PROP_TOPMOST, 1)  # Make the window topmost

    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Adding new User', cv2.WINDOW_NORMAL)  # Create a window that can be resized
    cv2.setWindowProperty('Adding new User', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Make the window fullscreen
    cv2.setWindowProperty('Adding new User', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)  # Set it back to normal but keeps the window resizable
    cv2.setWindowProperty('Adding new User', cv2.WND_PROP_TOPMOST, 1)  # Make the window topmost


    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

#Login functionality
from flask import Flask, request, render_template, redirect, url_for, session, flash
import os

app.secret_key = 'adminAdmin'  # Set a secret key for session management

USERNAME = 'admin'
PASSWORD = 'password'

@app.route('/login', methods=['GET', 'POST'])
def login():
    session.permanent = False

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == USERNAME and password == PASSWORD:  # Assuming you validate the credentials correctly
            session['user'] = username
            #flash('You were successfully logged in', 'info')
            return redirect(url_for('set_token'))
        else:
            flash('Invalid username or password', 'danger')
    # This line executes if it's a GET request or if the login failed
    return render_template('login.html')

@app.route('/set_token')
def set_token():
    # This route renders a template that includes the JavaScript
    # to set sessionStorage token indicating a valid login session
    return render_template('set_token.html')



@app.route('/logout')
def logout():
    session.clear()  # Clear the session data
    flash('You have been logged out.', 'info')  # Optionally, flash a message to the user
    return redirect(url_for('login'))  # Redirect back to the login page or another appropriate page

@app.route('/attendance_percentage')
def attendance_percentage():
    # Assuming extract_attendance() returns all necessary attendance information
    names, rolls, times, l = extract_attendance()
    total_sessions = calculate_total_sessions()  # You need to define this function
    
    # Dictionary to hold attendance percentages
    attendance_percentages = {}
    for name in set(names):  # Assuming names might have duplicates
        attended_sessions = names.count(name)
        attendance_percentage = (attended_sessions / total_sessions) * 100
        attendance_percentages[name] = attendance_percentage

    return render_template('attendance_percentage.html', attendance_percentages=attendance_percentages)


# Our main function which runs the Flask App
 # Set debug=False in a production environment
    #app.run(use_reloader=True)
if __name__ == '__main__':
    app.run(debug=True)
