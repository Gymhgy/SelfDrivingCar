# -*- coding: utf-8 -*-
from tkinter import *
from tkinter import ttk
import glob
from PIL import Image, ImageTk
from environment import Car, sand
from ai import Brain
import numpy as np
import torch
import cmath
import matplotlib.pyplot as plt

class App(Frame):
    def __init__(self, parent, *args, **kwargs):
        self.parent = parent
        Frame.__init__(self, parent, *args, **kwargs)
        
        self.saveBrain = ttk.Button(self, text = "Save Brain", command = self.save_brain)

        self.loadBrain = ttk.Button(self, text= "Load Brain", command = self.load_brain)
        
        self.startStop = ttk.Button(self, text = "Start", command = self.start)
        
        self.toggleFast = ttk.Button(self, text = "Fast Mode On", command = self.toggle_fast)
        self.fast = True
        brain = Brain(5,3)

        self.plot = ttk.Button(self, text = "Plot", command = self.plot)
        

        #Initialize map and road
        self.map = Map(self, Car(brain), width=700,height=500)
        self.RoadPalette = RoadPalette(self, width=360, height = 500)
        
        self.status = Label(self, text = "",anchor = "w", height=2, font=(None, 10))
        
        self.map.grid(row=0, column=0, columnspan=6, padx=5, pady=5)
        self.RoadPalette.grid(row=0, column=6, sticky="NSEW", padx=5, pady=5)         

        buttonrow = 1
        self.saveBrain.grid(row=buttonrow, column=0, pady=16)
        self.loadBrain.grid(row=buttonrow, column=1, pady=16)
        self.startStop.grid(row=buttonrow, column=2, pady=16)
        self.toggleFast.grid(row=buttonrow, column=3, pady=16)

        self.status.grid(row=2, column=0, columnspan=7, sticky="NSEW")
        
    def save_brain(self):
        torch.save({"state_dict": self.map.car.brain.policy_net.state_dict(),
            "optimizer" : self.map.car.brain.optimizer.state_dict(),
           }, "brain.pth")
        self.status.config(text = "Brain saved to brain.pth")
        #clear after 5 sec
        self.after(5000, lambda:self.status.config(text = ""))

    def load_brain(self):
        try:
            checkpoint = torch.load("brain.pth")
            self.map.car.brain.policy_net.load_state_dict(checkpoint["state_dict"])
            self.map.car.brain.optimizer.load_state_dict(checkpoint["optimizer"])
            self.status.config(text = "Brain loaded from brain.pth")
        except:
            self.status.config(text = "Could not load/find brain.pth")
        #clear after 5 sec
        self.after(5000, lambda:self.status.config(text = ""))

    def toggle_fast(self):
        if self.map.delay == 50:
            self.map.delay = 1
        else:
            self.map.delay = 50
        self.toggleFast.config(text="Fast Mode " + ["Off", "On"][self.map.delay==1])
    
    def start(self):
        self.saveBrain.config(state="disabled")
        self.loadBrain.config(state="disabled")
        self.startStop.config(text="Pause")
        self.startStop.config(command = self.stop)
        self.map.start()
    
    def stop(self):
        self.saveBrain.config(state="enabled")
        self.loadBrain.config(state="enabled")
        self.startStop.config(text="Start")
        self.startStop.config(command = self.start)
        self.map.stop()
        
class RoadPalette(ttk.Frame):

    def __init__(self, parent, **kwargs):
        self.parent = parent
        Frame.__init__(self, parent, **kwargs)
        #iterate through each file in the roads file
        i = 0
        for img in glob.glob("roads/*.png"):
            #Have to use PIL for png files
            orig = Image.open(img)
            orig_photo = ImageTk.PhotoImage(orig)
            im = orig.resize((80,80))
            photo = ImageTk.PhotoImage(im)
            #Save a copy of the non-resized photo so when the road is dragged over
            #the map, it can pass the non-resized photo to the map
            road = self.DraggableRoad(self, image=photo, orig_image = orig_photo, img_name = img)
            road.image = photo
            road.grid(column=i%2, row=i//2)
            i += 1
        
    class DraggableRoad(ttk.Label):
        def __init__(self, parent, orig_image, img_name, **kwargs):
            self.parent = parent
            ttk.Label.__init__(self, parent, **kwargs)
            self.orig_image = orig_image
            self.img_name = img_name
            
            self.bind("<ButtonPress-1>", self.start_drag)
            self.bind("<B1-Motion>", self.dragging)
            self.bind("<ButtonRelease-1>", self.released)
        
        def start_drag(self, event):
            self.startx = event.x
            self.starty = event.y
            #create a draggable we will manipulate
            self.draggable = Label(self.parent.parent, image=self.image)
            self.parent.parent.map.toggle_grid()


        def dragging(self, event):
            widget = event.widget
            x = self.parent.winfo_x() + self.winfo_x() + event.x - self.startx
            y = self.parent.winfo_y() + self.winfo_y() + event.y - self.starty
            self.draggable.place(x=x, y=y)
            
        #if released over the map, snap it to the map
        #also delete the draggable
        def released(self, event):
            self.draggable.destroy()
            x = self.winfo_containing(event.x_root, event.y_root)
            target = self.winfo_toplevel().nametowidget(x)
            if isinstance(target, Map):
                x = self.parent.winfo_x() + self.winfo_x() + event.x - target.winfo_x()
                y = self.parent.winfo_y() + self.winfo_y() + event.y - target.winfo_y()
                target.drop_road(self.orig_image,x, y, self.img_name)
            self.parent.parent.map.toggle_grid()

class Map(Canvas):
    def __init__(self, parent, car, **kwargs):
        self.parent = parent
        Canvas.__init__(self, parent, bg="yellow", **kwargs)
        self.img = ImageTk.PhotoImage(Image.open("grid.png"))
        self.grid_img = self.create_image(0,0, image=self.img, anchor="nw", state="hidden")
        self.bind("<ButtonPress-3>", self.delete_road)
        self._block = ImageTk.PhotoImage(Image.open("roads/Block.png"))
        for x in range(0,7):
            for y in range(0,5):
                self.tag_lower(self.create_image(x*100,y*100, image=self._block, anchor="nw"))
        self.delay = 1
        self.car = car
        self.car_img = self.create_polygon(50-16,50-8,50+16,50-8,50+16,50+8,50-16,50+8, fill="teal", outline="green")
        
    def drop_road(self, image, x, y, img_name):
        #don't drop anything at the top-left or bottom-right corner
        #those are the start and goal points
        if 0<=x<=100 and 0<=y<=100 or 600<=x<=700 and 400<=y<=500:
            return
        x-=x%100
        y-=y%100
        self.delete_at(x, y)
        self.tag_lower(self.create_image(x, y, image=image, anchor="nw"))

        img = Image.open(img_name)
        #get the red channel of the image, then check if it has a value that is 255
        #if yes set it to 1 else 0 because that means it is yellow aka sand
        #also transparent equals sand as well because the canvas has yellow background
        #r = red, a = alpha
        sand[y:y+100,x:x+100] = np.reshape(
            [int(r>0 or a==0) for (r,a) in zip(img.getdata(0),img.getdata(3))],
        (100,100))
        
    def delete_road(self, event):
        self.delete_at(event.x, event.y)
    
    def delete_at(self, x, y):
        #don't delete anything at the top-left or bottom-right corner
        #those are the start and goal points
        if 0<=x<=100 and 0<=y<=100 or 600<=x<=700 and 400<=y<=500:
            return
        x-=x%100
        y-=y%100
        #Get the road that is enclosed in a 100x100 box
        enclosed = self.find_enclosed(x,y,x+100,y+100)
        #don't delete the car
        for obj in enclosed:
            if obj != self.car_img:
                self.delete(obj)
        sand[y:y+100,x:x+100] = 1
        
    def toggle_grid(self):
        state = ["hidden", "normal"][self.itemcget(self.grid_img,"state")=="hidden"]
        self.itemconfig(self.grid_img, state = state)
        
    def update(self):
        finished = self.car.iterate()
        self.update_car_pos()

        if finished:
            self.car.reset()
        self._after_id = self.after(self.delay, self.update)
    
    def update_car_pos(self):
        x,y=self.car.position[0], self.car.position[1]
        #rotate and translation
        #http://web.archive.org/web/20200722205852/http://effbot.org/zone/tkinter-complex-canvas.htm
        newxy = []
        cangle = cmath.exp(self.car.rotation*np.pi/180*1j)
        center = complex(*self.car.position)
        for x,y in [[x-16,y-8],[x+16,y-8],[x+16,y+8],[x-16,y+8]]:
            v = cangle * (complex(x,y) - center) + center
            newxy.append(v.real)
            newxy.append(v.imag)
        self.coords(self.car_img, *newxy)
    
    def start(self):
        self._after_id = self.after(self.delay, self.update)
    
    def stop(self):
        self.after_cancel(self._after_id)


if __name__ == "__main__":
    root = Tk()
    root.minsize(width=1000, height=600)
    root.maxsize(width=1000, height=600)
    root.resizable(False, False)
    root.title("Self-Driving Car AI")
    App(root).pack(side="bottom", fill="both", expand=True)
    root.mainloop()        