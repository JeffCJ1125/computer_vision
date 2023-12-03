import numpy as np
import cv2 as cv
import glob

print(f"np version cur: {np.__version__}, test 1.26.0")
print(f"cv version cur: {cv.__version__}, test 4.6.0")
CHESSBOARD_ROW = 9
CHESSBOARD_COL = 6
print(f"chess boar row * col = {CHESSBOARD_ROW}*{CHESSBOARD_COL}")
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * CHESSBOARD_ROW, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_ROW, 0:CHESSBOARD_COL].T.reshape(-1, 2)
# my chess board square size is 20mm
objp = objp * 20
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
fns = glob.glob("*.jpg")
images = []
for i in range(len(fns)):
    fname = fns[i]
    img = cv.imread(fname)
    images.append(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(
        gray, (CHESSBOARD_ROW, CHESSBOARD_COL), None
    )
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (CHESSBOARD_ROW, CHESSBOARD_COL), corners2, ret)
        cv.imshow(f"img", img)
        cv.waitKey(500)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

with np.printoptions(formatter={"float": "{:0.3f}".format}):
    print(f"mtx \n {mtx}")

h, w = images[0].shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

with np.printoptions(formatter={"float": "{:0.3f}".format}):
    print(f"newcameramtx \n {newcameramtx}")

for i in range(len(images)):
    # undistort
    img = images[0]
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    output_name = fname[:-4] + "_calibresult.png"
    cv.imwrite(output_name, dst)
