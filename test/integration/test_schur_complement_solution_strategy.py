"""
Integration tests for the Schur Complement solution strategy.

We have the full Biot/THM equations in the matrix and mass 
(and energy) conservation and contact conditions in the fracture. 
TODO: Add 3d test.
"""
import numpy as np
import unittest, pdb

import porepy as pp
import porepy.models.contact_mechanics_biot_model as model
import porepy.models.thm_model as thm_model
import test.common.contact_mechanics_examples


class TestSchurComplement(unittest.TestCase):
    def _solve(self, setup):
        pp.run_time_dependent_model(setup, {"convergence_tol": 1e-6})

        gb = setup.gb

        nd = setup.Nd

        g2 = gb.grids_of_dimension(nd)[0]
        g1 = gb.grids_of_dimension(nd - 1)[0]

        d_m = gb.edge_props((g1, g2))
        d_1 = gb.node_props(g1)

        mg = d_m["mortar_grid"]

        u_mortar = d_m[pp.STATE][setup.mortar_displacement_variable]
        contact_force = d_1[pp.STATE][setup.contact_traction_variable]
        fracture_pressure = d_1[pp.STATE][setup.scalar_variable]

        displacement_jump_global_coord = (
            mg.mortar_to_slave_avg(nd=nd) * mg.sign_of_mortar_sides(nd=nd) * u_mortar
        )
        projection = d_m["tangential_normal_projection"]

        project_to_local = projection.project_tangential_normal(int(mg.num_cells / 2))
        u_mortar_local = project_to_local * displacement_jump_global_coord
        u_mortar_local_decomposed = u_mortar_local.reshape((nd, -1), order="F")

        contact_force = contact_force.reshape((nd, -1), order="F")

        return u_mortar_local_decomposed, contact_force, fracture_pressure

    def test_pull_north_positive_opening(self):
        uy = 0.001
        nx = 5
        nx = [nx, nx]
        setup = SetupContactMechanicsBiot(
            ux_south=0, uy_south=0, ux_north=0, uy_north=uy
        )
        setup.mesh_args = nx.copy()
        setup.simplex = False

        u_mortar, contact_force, fracture_pressure = self._solve(setup)
        setup_SC = SetupContactMechanicsBiot(
            ux_south=0,
            uy_south=0,
            ux_north=0,
            uy_north=uy,
            solution_strategy="schur_complement",
        )
        setup_SC.mesh_args = nx
        setup_SC.simplex = False
        u_mortar_SC, contact_force_SC, fracture_pressure_SC = self._solve(setup_SC)

        self.assertTrue(np.all(np.isclose(u_mortar_SC, u_mortar)))
        self.assertTrue(np.all(np.isclose(fracture_pressure_SC, fracture_pressure)))
        self.assertTrue(np.all(np.isclose(contact_force_SC, contact_force)))

    def test_multiple_time_steps(self):
        uy = 0.001
        nx = 5
        nx = [nx, nx]
        setup = SetupContactMechanicsBiot(
            ux_south=0, uy_south=0, ux_north=0, uy_north=uy
        )
        setup.mesh_args = nx.copy()
        setup.simplex = False
        setup.end_time *= 3
        u_mortar, contact_force, fracture_pressure = self._solve(setup)
        
        setup_SC = SetupContactMechanicsBiot(
            ux_south=0,
            uy_south=0,
            ux_north=0,
            uy_north=uy,
            solution_strategy="schur_complement",
        )
        setup_SC.mesh_args = nx
        setup_SC.simplex = False
        setup_SC.end_time *= 3
        u_mortar_SC, contact_force_SC, fracture_pressure_SC = self._solve(setup_SC)

        self.assertTrue(np.all(np.isclose(u_mortar_SC, u_mortar)))
        self.assertTrue(np.all(np.isclose(fracture_pressure_SC, fracture_pressure)))
        self.assertTrue(np.all(np.isclose(contact_force_SC, contact_force)))

    def test_multiple_time_steps_thm(self):
        uy = 0.001
        nx = 15
        nx = [nx, nx]
        setup = SetupTHM(
            ux_south=0, uy_south=0, ux_north=0, uy_north=uy
        )
        setup.mesh_args = nx.copy()
        setup.simplex = False
        setup.end_time *= 3
        u_mortar, contact_force, fracture_pressure = self._solve(setup)
        
        setup_SC = SetupTHM(
            ux_south=0,
            uy_south=0,
            ux_north=0,
            uy_north=uy,
            solution_strategy="schur_complement",
        )
        setup_SC.mesh_args = nx
        setup_SC.simplex = False
        setup_SC.end_time *= 3
        u_mortar_SC, contact_force_SC, fracture_pressure_SC = self._solve(setup_SC)

        self.assertTrue(np.all(np.isclose(u_mortar_SC, u_mortar)))
        self.assertTrue(np.all(np.isclose(fracture_pressure_SC, fracture_pressure)))
        self.assertTrue(np.all(np.isclose(contact_force_SC, contact_force)))

class SetupContactMechanicsBiot(
    test.common.contact_mechanics_examples.ProblemDataTime, model.ContactMechanicsBiot
):
    def __init__(
        self,
        ux_south,
        uy_south,
        ux_north,
        uy_north,
        solution_strategy=None,
        source_value=0,
    ):

        self.mesh_args = {
            "mesh_size_frac": 0.5,
            "mesh_size_min": 0.023,
            "mesh_size_bound": 0.5,
        }
        params = {"solution_strategy": solution_strategy}
        super().__init__(params)

        self.ux_south = ux_south
        self.uy_south = uy_south
        self.ux_north = ux_north
        self.uy_north = uy_north
        self.scalar_source_value = source_value


class SetupTHM(
    test.common.contact_mechanics_examples.ProblemDataTime, thm_model.THM
):
    def __init__(
        self,
        ux_south,
        uy_south,
        ux_north,
        uy_north,
        solution_strategy=None,
        source_value=0,
    ):

        self.mesh_args = {
            "mesh_size_frac": 0.5,
            "mesh_size_min": 0.023,
            "mesh_size_bound": 0.5,
        }
        params = {"solution_strategy": solution_strategy}
        super().__init__(params)

        self.ux_south = ux_south
        self.uy_south = uy_south
        self.ux_north = ux_north
        self.uy_north = uy_north
        self.scalar_source_value = source_value

if __name__ == "__main__":
#    TestSchurComplement().test_multiple_time_steps_thm()
    unittest.main()
