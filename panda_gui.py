import rospy
from std_msgs.msg import Float32MultiArray
from tkinter import *
from tkinter import ttk

class JointPublisherGUI:
    def __init__(self, master):
        self.master = master
        master.title("Panda Joint Controller")

        self.joint_names = [f"panda_joint{i+1}" for i in range(7)]
        self.joint_vars = [DoubleVar(value=0.0) for _ in range(7)]
        self.sliders = []

        self.auto_publish = False

        # Sliders
        for i, (name, var) in enumerate(zip(self.joint_names, self.joint_vars)):
            label = Label(master, text=name)
            label.grid(row=i, column=0)

            slider = Scale(
                master, from_=-3.14, to=3.14, resolution=1e-8,
                orient=HORIZONTAL, variable=var, length=300
            )
            slider.grid(row=i, column=1, padx=5, pady=2)
            self.sliders.append(slider)

            entry = Entry(master, width=16)
            entry.grid(row=i, column=2, padx=5)
            entry.insert(0, f"{var.get():.8f}")
            slider.entry = entry

            # Entry-to-slider update
            def make_callback(v, e):
                def on_change(_):
                    try:
                        val = float(e.get())
                        v.set(val)
                        if self.auto_publish:
                            self.publish_joints()
                    except:
                        pass
                return on_change
            entry.bind("<Return>", make_callback(var, entry))

            # Mouse wheel micro-adjust
            def make_mousewheel_callback(v, s):
                def on_mousewheel(event):
                    delta = 1 if event.delta > 0 else -1
                    step = 0.0001
                    new_val = v.get() + delta * step
                    new_val = max(s.cget("from"), min(new_val, s.cget("to")))
                    v.set(new_val)
                    s.entry.delete(0, "end")
                    s.entry.insert(0, f"{new_val:.8f}")
                    if self.auto_publish:
                        self.publish_joints()
                return on_mousewheel
            slider.bind("<MouseWheel>", make_mousewheel_callback(var, slider))
            slider.bind("<Button-4>", make_mousewheel_callback(var, slider))  # Linux up
            slider.bind("<Button-5>", make_mousewheel_callback(var, slider))  # Linux down

        # Text input field for batch assignment
        self.input_label = Label(master, text="Set All (comma-separated):")
        self.input_label.grid(row=7, column=0, columnspan=1, sticky=W, pady=(10, 0))

        self.input_entry = Entry(master, width=50)
        self.input_entry.grid(row=7, column=1, columnspan=2, pady=(10, 0))
        self.input_entry.bind("<Return>", self.set_from_text_input)

        # Toggle auto publish button
        self.toggle_btn = Button(master, text="Start Auto Publish", command=self.toggle_publish)
        self.toggle_btn.grid(row=8, column=0, pady=10)

        # Manual publish button
        self.publish_btn = Button(master, text="Publish Once", command=self.publish_joints)
        self.publish_btn.grid(row=8, column=1, pady=10)

    def toggle_publish(self):
        self.auto_publish = not self.auto_publish
        self.toggle_btn.config(
            text="Stop Auto Publish" if self.auto_publish else "Start Auto Publish"
        )
        if self.auto_publish:
            self.publish_joints()

    def set_from_text_input(self, event=None):
        try:
            values = [float(v.strip()) for v in self.input_entry.get().split(",")]
            if len(values) != 7:
                raise ValueError("Need 7 values")
            for var, val, slider in zip(self.joint_vars, values, self.sliders):
                val = max(slider.cget("from"), min(val, slider.cget("to")))
                var.set(val)
                slider.entry.delete(0, "end")
                slider.entry.insert(0, f"{val:.8f}")
            if self.auto_publish:
                self.publish_joints()
        except Exception as e:
            print(f"[Input Error] {e}")

    def publish_joints(self):
        joint_values = [var.get() for var in self.joint_vars]
        msg = Float32MultiArray(data=joint_values)
        pub.publish(msg)
        print("Publishing:", joint_values)

if __name__ == "__main__":
    rospy.init_node("joint_gui_publisher")
    pub = rospy.Publisher("/pick", Float32MultiArray, queue_size=10)
    root = Tk()
    gui = JointPublisherGUI(root)
    root.mainloop()
