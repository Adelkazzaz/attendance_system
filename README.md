# Face Recognition App

The Face Recognition App is a simple Python application that utilizes facial recognition technology to perform various tasks such as attendance tracking and student identification. It allows users to interact with a graphical user interface to control the application's functionality.

## Features

- **Real-time Face Detection**: Utilizes computer vision techniques to detect faces in real-time from a webcam feed.
- **Facial Recognition**: Recognizes known faces based on pre-trained face encodings and labels them accordingly.
- **Attendance Tracking**: Allows users to start and stop attendance tracking sessions, and copy the list of attendees' names to the clipboard.
- **Adding New Students**: Provides functionality to add new students by capturing their images from video files and associating them with unique identifiers.
- **Switching Cameras**: Enables users to switch between different available cameras for face detection.

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/Face_Recognition_App.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python run.py
   ```

## Usage

1. Upon launching the application, the webcam feed will be displayed, and the application will be ready to detect and recognize faces.
2. Use the provided buttons to control various functionalities such as starting/stopping attendance tracking, adding new students, switching cameras, and quitting the application.
3. To add a new student, click on the "Add Student" button and select a video file containing the student's images. Enter the student's name and ID, then click "OK" to capture the best frame for facial recognition.
4. Start and stop attendance tracking sessions using the corresponding buttons. The list of attendees' names will be displayed in real-time.
5. Click on the "Copy Names" button to copy the list of attendees' names to the clipboard.

## Dependencies

- OpenCV
- Pillow
- numpy

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize this README file according to your project's specific details and requirements. Let me know if you need further assistance!