import sys
from PyQt5 import QtWidgets
from ui.face_emotion import Ui_Form
from handlers.event_handlers import EventHandlers

class MainApp(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.handlers = EventHandlers(self)
        self.setup_connections()

    def setup_connections(self):
        self.pushButton.clicked.connect(self.handlers.on_pushButton_clicked)
        self.pushButton_2.clicked.connect(self.handlers.on_pushButton_2_clicked)
        self.pushButton_3.clicked.connect(self.handlers.on_pushButton_3_clicked)
        self.pushButton_4.clicked.connect(self.handlers.on_pushButton_4_clicked)
        self.pushButton_5.clicked.connect(self.handlers.on_pushButton_5_clicked)
        self.pushButton_6.clicked.connect(self.handlers.on_pushButton_6_clicked)
        self.pushButton_7.clicked.connect(self.handlers.on_pushButton_7_clicked)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())