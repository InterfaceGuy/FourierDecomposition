# Code to draw cool things using the math of complex Fourier-series
# This is an updated version of the code from youtuber Theorem of Beethoven as seen here:
# Theorem of Beethoven link:    https://www.youtube.com/watch?v=2tTshwWTEic
# Adapted from CairoManim to ManimCE
# Inspired by brilliant math youtuber 3Blue1Brown, creator of the Manim Python library:
# 3lue1Brown link:              https://www.youtube.com/watch?v=r6sGWTCMz2k

from manim import *
import numpy as np
from svgpathtools import svg2paths

# config.use_opengl_renderer = True

class FourierSceneAbstract(ZoomedScene):
    def __init__(self):
        super().__init__()
        self.fourier_symbol_config = {
            "stroke_width": 1,
            "fill_opacity": 1,
            "height": 4,
        }
        self.vector_config = {
            "buff": 0,
            "max_tip_length_to_length_ratio": 0.25,
            "tip_length": 0.15,
            "max_stroke_width_to_length_ratio": 10,
            "stroke_width": 1.4
        }
        self.circle_config = {
            "stroke_width": 1,
            "stroke_opacity": 0.3,
            "color": WHITE
        }
        self.n_vectors = 40
        self.cycle_seconds = 5
        self.parametric_func_step = 0.001   
        self.drawn_path_stroke_width = 5
        self.drawn_path_interpolation_config = [0, 1]
        self.path_n_samples = 1000
        self.freqs = list(range(-self.n_vectors // 2, self.n_vectors // 2 + 1, 1))
        self.freqs.sort(key=abs)

    def setup(self):
        super().setup()
        self.vector_clock = ValueTracker()
        self.slow_factor_tracker = ValueTracker(0)
        self.add(self.vector_clock)

    def start_vector_clock(self):           # This updates vector_clock to follow the add_updater parameter dt
        self.vector_clock.add_updater(
            lambda t, dt: t.increment_value(dt * self.slow_factor_tracker.get_value() / self.cycle_seconds)
        )

    def stop_vector_clock(self):
        self.vector_clock.remove_updater(self.start_vector_clock)

    def get_fourier_coefs(self, path):
        dt = 1 / self.path_n_samples
        t_range = np.arange(0, 1, dt)

        points = np.array([
            path.point_from_proportion(t)
            for t in t_range
        ])
        complex_points = points[:, 0] + 1j * points[:, 1]

        coefficients = [
            np.sum(np.array([
                c_point * np.exp(-TAU * 1j * freq * t) * dt
                for t, c_point in zip(t_range, complex_points)
                ]))
            for freq in self.freqs
        ]
        return coefficients

    def get_fourier_vectors(self, path):
        coefficients = self.get_fourier_coefs(path)
        
        vectors = VGroup()
        v_is_first_vector = True
        for coef, freq in zip(coefficients,self.freqs):
            v = Vector([np.real(coef), np.imag(coef)], **self.vector_config)
            if v_is_first_vector:
                center_func = VectorizedPoint(ORIGIN).get_location # Function to center position at tip of last vector
                v_is_first_vector = False
            else:
                center_func = last_v.get_end
            v.center_func = center_func
            last_v = v
            v.freq = freq
            v.coef = coef
            v.phase = np.angle(coef)
            v.shift(v.center_func()-v.get_start())
            v.set_angle(v.phase)
            vectors.add(v)
        return vectors

    def update_vectors(self, vectors):
            for v in vectors:
                time = self.vector_clock.get_value()
                v.shift(v.center_func()-v.get_start())
                v.set_angle(v.phase + time * v.freq * TAU)  # NOTE Rotate() did not work here for unknown reason, probably related to how manin handles updaters
              
    def get_circles(self, vectors):
        circles = VGroup()
        for v in vectors:
            c = Circle(radius = v.get_length(), **self.circle_config)
            c.center_func = v.get_start
            c.move_to(c.center_func())
            circles.add(c)
        return circles

    def update_circles(self, circles):
        for c in circles:
            c.move_to(c.center_func())
            
    def get_drawn_path(self, vectors):    # TODO Find out application of None, is for placeholder, may be how keyword argument default is set

        def fourier_series_func(t):
            fss = np.sum(np.array([
                v.coef * np.exp(TAU * 1j * v.freq * t)
                for v in vectors
            ]))
            real_fss = np.array([np.real(fss), np.imag(fss), 0])
            return real_fss
        
        t_range = np.array([0, 1, self.parametric_func_step])
        vector_sum_path = ParametricFunction(fourier_series_func, t_range = t_range)
        broken_path = CurvesAsSubmobjects(vector_sum_path)
        broken_path.stroke_width = 0
        broken_path.start_width = self.drawn_path_interpolation_config[0]
        broken_path.end_width = self.drawn_path_interpolation_config[1]
        return broken_path

    def update_path(self, broken_path):
        alpha = self.vector_clock.get_value()
        n_curves = len(broken_path)
        alpha_range = np.linspace(0, 1, n_curves)
        for a, subpath in zip(alpha_range, broken_path):
            b = (alpha - a)
            if b < 0:
                width = 0
            else:
                width = self.drawn_path_stroke_width * interpolate(broken_path.start_width, broken_path.end_width, (1 - (b % 1)))
            subpath.set_stroke(width=width)

class FourierScene(FourierSceneAbstract):
    def __init__(self):
        super().__init__()

    def get_svg_path(self, file_path):
        paths, attributes = svg2paths(file_path)
        return VMobject().set_points_smoothly([complex_to_R3(complex(*z)) for z in paths[0].points()])

    def construct(self):
        # SVG path to draw
        svg_path = self.get_svg_path("path/to/your/svg/file.svg")  # Replace with your SVG file path
        group = VGroup(svg_path)

        # Fourier series for the SVG path
        vectors = self.get_fourier_vectors(svg_path)
        circles = self.get_circles(vectors)
        drawn_path = self.get_drawn_path(vectors).set_color(RED)

        all_mobs = group

        # Camera updater
        last_vector = vectors[-1]

        def follow_end_vector(camera): 
            camera.move_to(last_vector.get_end())

        # Scene start
        self.wait(1)
        self.play(
            *[GrowArrow(arrow) for arrow in vectors],
            *[Create(circle) for circle in circles],
            run_time=2.5,
        )

        # Add objects to scene
        self.add(
            vectors,
            circles,
            drawn_path.set_stroke(width=0),
        )

        # Camera move
        self.play(self.camera.frame.animate.scale(0.3).move_to(last_vector.get_end()), run_time=2)

        # Add updaters and start vector clock
        self.camera.frame.add_updater(follow_end_vector)
        vectors.add_updater(self.update_vectors)
        circles.add_updater(self.update_circles)
        drawn_path.add_updater(self.update_path)
        self.start_vector_clock()

        self.play(self.slow_factor_tracker.animate.set_value(1), run_time=0.5 * self.cycle_seconds)
        self.wait(1 * self.cycle_seconds)

        # Move camera out
        self.camera.frame.remove_updater(follow_end_vector)
        self.play(
            self.camera.frame.animate.set_width(all_mobs.width * 1.5).move_to(all_mobs.get_center()),
            run_time=1 * self.cycle_seconds,
        )
        self.wait(0.8 * self.cycle_seconds)
        self.play(self.slow_factor_tracker.animate.set_value(0), run_time=0.5 * self.cycle_seconds)
        
        # Remove updaters so can animate
        self.stop_vector_clock()
        drawn_path.clear_updaters()
        vectors.clear_updaters()
        circles.clear_updaters()

        self.play(
            *[Uncreate(vmobject) for vmobject in vectors + circles],
            FadeOut(drawn_path),
            FadeIn(svg_path),
            run_time=2.5,
        )

        self.wait(3)
