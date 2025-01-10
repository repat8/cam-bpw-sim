// (c) Répai Attila, 2023

// SC.1.0 - initial version
// SC.1.1 - marker at each 45°, starting from the start of the nominal signal

module bp_simulator_cam(
    signal_points,
    r,
    r_rim,
    r_follower,
    name,
    amplitude,
    d_shaft,
    rotation,
    cam_version  // Needs to be incremented in the Python module and passed as param
) {
    h_chamfer_top = 0.4;
    h_chamfer_bottom = 0.8;
    surface_width = 2;
    l_shaft = 10;
    z_top_surface = surface_width + h_chamfer_bottom + h_chamfer_top;
    difference() {
        union() {
            translate([0, 0, h_chamfer_bottom])
            rotate([180, 0, 0])
            linear_extrude(h_chamfer_bottom, scale=(r - h_chamfer_bottom) / r)
            rotate([180, 0, 0])
            polygon(signal_points);

            translate([0, 0, h_chamfer_bottom])
            linear_extrude(surface_width)
            polygon(signal_points);

            translate([0, 0, surface_width + h_chamfer_bottom])
            linear_extrude(h_chamfer_top, scale=(r - h_chamfer_top) / r)
            polygon(signal_points);

            difference() {
                cylinder(r=r_rim, h=6, $fn = 4 * 360);
                translate([0, 0, h_chamfer_bottom+surface_width+h_chamfer_top])
                cylinder(r=r_rim-1, h=l_shaft, $fn = 4 * 360);
            }

            cylinder(d=d_shaft + 4, h=l_shaft);

            for (rot = [0:45:360]) {
                rotate([0, 0, rot])
                translate([r_rim - 5, -0.2, z_top_surface])
                cube([5, 0.4, 0.4]);
            }

            translate([0, -r / 2, z_top_surface])
            linear_extrude(0.4)
            text(str(amplitude, "/", r_follower), halign="center", valign="center", size=4);

            translate([0, r / 2, z_top_surface])
            linear_extrude(0.4)
            text(name, halign="center", valign="center", size=4);

            translate([-r / 2, 0, z_top_surface])
            rotate([0, 0, 90])
            linear_extrude(0.4)
            text(cam_version, halign="center", valign="center", size=4);

            translate([r / 2, 0, z_top_surface])
            rotate([0, 0, 90])
            linear_extrude(0.4)
            text(rotation == "ccw" ? ">" : "<", halign="center", valign="center", size=4);
        }
        cylinder(d=d_shaft, h=l_shaft);
        translate([0, 0, l_shaft-3.5]) rotate([90, 0, 0]) cylinder(d=1.8, h=d_shaft + 5);
        linear_extrude(h_chamfer_bottom, scale=(d_shaft - 2*h_chamfer_bottom) / d_shaft)
        circle(d=d_shaft+2*h_chamfer_bottom);
    }
}
