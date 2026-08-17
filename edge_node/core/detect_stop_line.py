 def detect_stop_line(self, show_stages=False):
        """
        Detect and return the y-coordinate of the stop line.
        
        This method uses computer vision techniques to detect the stop line in the video:
        1. Converts frame to grayscale
        2. Applies adaptive thresholding
        3. Uses morphological operations to clean up the image
        4. Finds contours and filters for rectangular shapes
        5. Selects the most likely stop line based on position relative to traffic light
        
        Args:
            show_stages (bool): If True, displays intermediate processing stages
            
        Returns:
            int: Y-coordinate of the stop line, or 0 if not found
        """
        self.logger.info("Detecting stop line position...")
        
        if self.traffic_light_coords is None:
            self.logger.error("Traffic light coordinates not detected")
            return 0
            
        xlight, ylight, wlight, hlight = self.traffic_light_coords
        
        cap = cv2.VideoCapture(self.input_path)
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            self.logger.error("Could not read frame for stop line detection")
            return 0
            
        frame = cv2.resize(frame, (VIDEO_PROCESSING['RESIZE_WIDTH'], 
                                 VIDEO_PROCESSING['RESIZE_HEIGHT']))
        if show_stages:
            cv2.imshow('Original Frame', frame)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        temp = frame.copy()
        temp2 = frame.copy()
        
        # Convert image to grayscale
        grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        th = cv2.adaptiveThreshold(
            grayscaled, 250, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY,
            STOP_LINE_DETECTION['ADAPTIVE_THRESH_BLOCK_SIZE'],
            STOP_LINE_DETECTION['ADAPTIVE_THRESH_C']
        )
        kernel = np.ones((3, 3), np.uint8)
        
        # Erode and dilate to filter noise
        th = cv2.erode(th, kernel, iterations=STOP_LINE_DETECTION['ERODE_ITERATIONS'])
        th = cv2.dilate(th, kernel, iterations=STOP_LINE_DETECTION['DILATE_ITERATIONS'])
        
        if show_stages:
            cv2.imshow('Threshold Image', th)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        # Find contours in the processed image
        contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        
        all_contours = []
        
        # Filter contours to find suitable rectangles
        for i, contour in enumerate(contours):
            if (cv2.contourArea(contour) > STOP_LINE_DETECTION['MIN_CONTOUR_AREA'] and 
                len(contour) < STOP_LINE_DETECTION['MAX_CONTOUR_POINTS']):
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 
                                        STOP_LINE_DETECTION['APPROX_POLY_EPSILON'] * peri, 
                                        True)
                if len(approx) == 4:  # Find contours with 4 sides (rectangles)
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.drawContours(frame, contours, i, UI_DISPLAY['COLORS']['GREEN'], 3)
                    cv2.rectangle(temp, (x, y), (x+w, y+h), UI_DISPLAY['COLORS']['RED'], 2)
                    all_contours.append((x, y, w, h))
        
        cv2.drawContours(temp2, contours, -1, UI_DISPLAY['COLORS']['GREEN'], 3)
        
        if show_stages:
            cv2.imshow('All Contours', temp2)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.imshow('Detected Rectangle Contours', frame)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.imshow('Bounding Boxes', temp)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        min_index = 0
        min_distance = float('inf')
        
        # delete the rectangle that upper than the traffic light
        all_contours = [rect for rect in all_contours if rect[1] > ylight + hlight]
        if not all_contours:
            self.logger.warning("No suitable rectangles found above the traffic light")
            cap.release()
            return 0
        
        # Find the rectangle closest to the traffic light (possible stop line)
        for i, rect in enumerate(all_contours):
            x, y, w, h = rect
            if ylight + wlight < y:
                cv2.line(temp, (xlight, ylight), (x, y), UI_DISPLAY['COLORS']['RED'], 2)
                distance = ((x-xlight)**2 + (y-ylight)**2)**0.5
                if distance < min_distance:
                    min_distance = distance
                    min_index = i
        
        if all_contours:
            x, y, w, h = all_contours[min_index]
            if show_stages:
                cv2.imshow('Distance Visualization', temp)
                cv2.waitKey()
                cv2.destroyAllWindows()
                cv2.line(temp, (0, y), (1300, y), UI_DISPLAY['COLORS']['BLACK'], 
                        UI_DISPLAY['LINE_THICKNESS'], cv2.LINE_AA)
                cv2.imshow('Stop Line', temp)
                cv2.waitKey()
                cv2.destroyAllWindows()
            cap.release()
            self.logger.info(f"Stop line detected at y={y}")
            return y
        
        cap.release()
        self.logger.warning("No stop line detected")
        return 0