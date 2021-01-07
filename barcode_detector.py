import cv2
import numpy as np
import pandas as pd
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
vid_cod = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("video.avi", vid_cod, 30.0, (640,480))

marketim_df = pd.read_csv('Marketim-Product.csv')




while True:
    success, frame = cap.read()
    for barcode in decode(frame):
        barcode_no = barcode.data.decode('utf-8')

        if barcode_no in marketim_df['_ywbc_barcode_value'].values:
            output = 'identified'
            outputColor = (0, 255, 0)
        else:
            output = 'Not-identified'
            outputColor = (0, 0, 255)

        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1,1,2))
        pts2 = barcode.rect
        cv2.polylines(frame, [pts],True, outputColor, 5)
        cv2.putText(frame, output, (pts2[0], pts2[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, outputColor, 4 )
        

    cv2.imshow('Result', frame)
    out.write(frame)
    key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
	    break
# close the already opened camera
cap.release()
# close the already opened file
out.release()
# close the window and de-allocate any associated memory usage
cv2.destroyAllWindows()      
