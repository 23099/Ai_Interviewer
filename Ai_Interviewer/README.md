Instructions for running the project:

Steps:
    1. go to "dependencies/python-3.10.0-amd64.exe" and install the python 3.10, during installation MAKE SURE to check the "Add Python 3.10 To Path" option
    2. open powershell in the root directory and run the following commands one after the other
        py -3.10 -m venv hci_env
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
        hci_env\Scripts\activate
        python -m pip install --upgrade pip setuptools wheel
        pip install -r requirements.txt
    3. run the project by running the following command
        python hci_detection_Group30.py