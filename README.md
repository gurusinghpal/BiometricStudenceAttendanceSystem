# Biometric Student Attendance System

## Introduction
The Biometric Student Attendance System is a web-based application that utilizes facial recognition technology to record and monitor student attendance. This system provides a secure, efficient, and user-friendly method for managing student attendance, eliminating the need for traditional manual entry methods.

## Features
- **Facial Recognition Attendance:** Automatically marks attendance by recognizing and verifying students' faces.
- **Real-Time Updates:** Attendance data is updated in real-time and can be viewed immediately by educators.
- **Reports Generation:** Generate attendance reports for individual students, classes, or custom date ranges.
- **Secure Access:** Role-based access control for administrators, teachers, and students.
- **Responsive Design:** Compatible with various devices, ensuring accessibility on desktops, laptops, tablets, and mobile phones.

## Tech Stack
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python (Flask as the web framework)
- **Facial Recognition:** OpenCV and face_recognition library
- **Database:** SQLite

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Node.js and npm (for frontend dependencies)

### Installation

1. **Clone the repository**
    ```bash
    git clone [https://github.com/gurusinghpal/BiometricStudenceAttendanceSystem](https://github.com/gurusinghpal/BiometricStudenceAttendanceSystem).git
    cd BiometricAttendanceSystem
    ```

2. **Set up the Python environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Install frontend dependencies**
    ```bash
    cd frontend
    npm install
    ```

4. **Start the backend server**
    ```bash
    cd ..
    python app.py
    ```

5. **Launch the frontend**
    ```bash
    cd frontend
    npm start
    ```

6. **Navigate to `http://localhost:3000` in your web browser to view the application.**

## Usage
To use the system, log in as an administrator to enroll new students. Use the provided interface to upload student images and enter details. Teachers can log in to view and monitor attendance records in real-time.

## Contributing
Contributions are welcome! If you have suggestions for improvements or bug fixes, please feel free to fork the repository and submit a pull request.

1. **Fork the Project**
2. **Create your Feature Branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your Changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the Branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
- Project Link: [https://github.com/gurusinghpal/BiometricStudenceAttendanceSystem](https://github.com/gurusinghpal/BiometricStudenceAttendanceSystem)
- Your Email: gurusingh2585@youremail.com

---

We look forward to seeing how the Biometric Student Attendance System can make classroom management more efficient and secure!

![Face Recognition Based Attendance System](ss.png)
