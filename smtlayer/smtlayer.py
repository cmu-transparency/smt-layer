import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

import warnings
import z3

class Z3SolverOp(torch.autograd.Function):
	"""
		A torch.autograd.Function that takes a batch of floating-point
		inputs, converts them to a set of z3 literals, and asserts them
		in a z3 solver. The resulting model is then converted back to
		floating-point, and returned as the output.
	"""

	@staticmethod
	def _float_to_bool(float_v: float, var: z3.BoolRef) -> z3.BoolRef:
		"""
			Converts a floating-point value to a z3 literal.

			Arguments:
				float_v: Floating-point value
				var:     z3 BoolRef

			Returns: If `float_v` is negative, then `var` negated, otherwise
				`var`.
		"""
		
		return z3.Not(var) if float_v < 0 else var
	
	@staticmethod
	def _z3_var_constraints(v: float, z3_vars: list[z3.BoolRef]) -> list[z3.BoolRef]:
		"""
			Converts a floating-point vector to a set of z3 literals.

			Arguments:
				v:        Floating-point vector
				z3_vars:  List of z3 variables

			Returns: A list of z3 literals, one for each element of `v`.
		"""

		return Z3SolverOp._floatvec_to_cons(v, z3_vars)

	@staticmethod
	def _floatvec_to_cons(v: list[float], vs: list[z3.BoolRef]) -> list[z3.BoolRef]:
		"""
			Converts a floating-point vector to a set of z3 literals.

			Arguments:
				v:        Floating-point vector
				z3_vars:  List of z3 variables
			
			Returns: A list of z3 literals, one for each element of `v`.
		"""
		return [Z3SolverOp._float_to_bool(v[i], vs[i]) for i in range(len(vs))]

	@staticmethod
	def _get_input_output_vars(mask: torch.Tensor, z3_vars: list[z3.BoolRef], var_ids: list[z3.BoolRef]):
		"""
			Extracts the input and output z3 variables from a mask.

			Arguments:
				mask:     Mask of input and output variables
				z3_vars:  List of z3 variables
				var_ids:  List of z3 variables used for unsat core tracking
			
			Returns: A tuple of lists of z3 variables: (input, output, input_ids)
		"""
		input_idx = torch.nonzero(mask).flatten().detach().cpu().numpy().tolist()
		output_idx = torch.nonzero(1.-mask).flatten().detach().cpu().numpy().tolist()
		z3_inputs = [z3_vars[i] for i in input_idx]
		if var_ids is not None:
			input_ids = [z3.Bool('var{}'.format(i)) for _, i in enumerate(input_idx)]
		else:
			input_ids = None
		z3_outputs = [z3_vars[i] for i in output_idx]

		return z3_inputs, z3_outputs, input_ids

	@staticmethod
	def _assert_and_check(
		x: torch.Tensor, 
		mask: torch.Tensor, 
		z3_vars: list[z3.BoolRef], 
		var_ids: list[z3.BoolRef],
		solver: z3.Solver,
		opt: z3.Optimize, 
		do_maxsat: bool=False, 
		return_cores: bool=False, 
		is_default_mask: bool=False
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""
			Assert a batch of floating-point inputs in a z3 solver,
			and check satisfiability. This is the essential forward pass
			of the layer.

			Arguments:
				x:          Floating-point input
				mask:       Mask of input and output variables
				z3_vars:    List of z3 variables, length matching mask
				var_ids:    List of z3 variables used for unsat core tracking
				solver:     z3 solver
				opt:        z3 optimizer
				do_maxsat:  Whether to use maxsat. Defaults to false, but if
							true, then the maxsat engine is used instead of sat
				return_cores: Whether to enable unsat core tracking
				is_default_mask: Whether the mask is a default mask that applies
								 to the entire batch, or a per-example mask
			
			Returns: If the input given by the assignment is sat, then return 
				the satisfying assignment as a floating-point tensor. If unsat, 
				return the default value of all-0's.
		"""
				
		x_flip = torch.ones_like(x)
		y = []
		cores = []

		if is_default_mask:
			z3_inputs, z3_outputs, input_ids = Z3SolverOp_MaxSat._get_input_output_vars(mask[0], z3_vars, var_ids)
			n_outputs = len(z3_outputs)
		
		# input is batched; we should update this to a parallel implementation,
		# as each iteration is independent.
		for i, xi in enumerate(x):
			core = None

			if not is_default_mask:
				z3_inputs, z3_outputs, input_ids = Z3SolverOp_MaxSat._get_input_output_vars(mask[i], z3_vars, var_ids)
				n_outputs = len(z3_outputs)
			
			# save the current state of the solver
			solver.push()
			opt.push()
			
			# convert the floating-point input to a set of z3 literals,
			# and assert each one in the solver
			li = Z3SolverOp._floatvec_to_cons(xi[mask[i]==1], z3_inputs)
			probits = F.softmax(torch.abs(xi[mask[i]==1]), dim=-1)
			for j, xcon in enumerate(li):
				if not False: # model._is_fixed[z3_inputs[j]]:
					opt.add_soft(xcon, probits[j].item())
				else:
					opt.add(xcon)
				solver.add(xcon)
			
			# check satisfiability of the current solver state
			if solver.check() == z3.sat:
				# it's sat, so read out the satisfying assignment,
				# convert to float, and save the tensor
				m = solver.model()
				outputs = [z3.is_true(m.eval(v, model_completion=True)) for v in z3_outputs]
				y.append(2*(torch.Tensor(outputs).unsqueeze(0).float().to(x.device)-0.5))
			else:
				if do_maxsat:
					opt.check()
					m = opt.model()
					outputs = [z3.is_true(m.eval(v, model_completion=True)) for v in z3_outputs]
					y.append(2*(torch.Tensor(outputs).unsqueeze(0).float().to(x.device)-0.5))
					indices_minus = [
						j 
						for j, xcon in enumerate(li) 
						if z3.is_false(m.eval(xcon, model_completion=True))
					]
					x_flip[i, indices_minus] = -1*x_flip[i, indices_minus]
				else:
					# it's unsat, so get the unsat core if requested,
					# and return the default value of all-0's
					y.append(torch.zeros((1, n_outputs)).float().to(x.device))
			
			# recover the previous solver state
			opt.pop()
			solver.pop()
			cores.append(core)
		
		# construct the result tensor, and convert from
		# {0,1} to {-1,1}
		try:
			y = torch.concat(y)
		except:
			y = torch.nested.to_padded_tensor(torch.nested.nested_tensor(y), 2)
		
		if return_cores:
			return y, cores
		else:
			return y, x_flip

	@staticmethod
	def _assert_clauses(theory: list[z3.BoolRef], clause_ids: list[z3.BoolRef], solver: z3.Solver) -> None:
		"""
			Assert a list of clauses in a z3 solver.

			Arguments:
				theory:     List of z3 clauses
				clause_ids: List of z3 variables used for unsat core tracking
				solver:     z3 solver
			
			Returns: None
		"""

		for i, cl in enumerate(theory):
			if clause_ids is not None:
				solver.assert_and_track(cl, clause_ids[i])
			else:
				solver.add(cl)

	@staticmethod
	def forward(
		ctx: torch.autograd.function._ContextMethodMixin,
		x: torch.Tensor,
		do_maxsat_forward: bool, 
		mask: torch.Tensor,
		grad_scaling: float,
		model: torch.nn.Module,
		is_default_mask
	) -> torch.Tensor:
		"""
			Forward pass of the layer. Takes a batch of floating-point inputs,
			and returns a batch of floating-point outputs.

			Arguments:
				ctx:        		Context object
				x:          		Floating-point input
				do_maxsat_forward: 	Whether to use maxsat. Defaults to false, but if
									true, then the maxsat engine is used instead of sat
				mask:       		Mask of input and output variables
				grad_scaling: 		Scaling factor for the gradient
				model:      		The layer itself 
				is_default_mask: 	Whether the mask is a default mask that applies
									to the entire batch, or a per-example mask
			
			Returns: A batch of floating-point outputs.
		"""

		in_device = x.device
		
		ctx.z3_vars = model._z3_vars
		ctx.var_ids = model._var_ids
		ctx.clause_ids = model._clause_ids
		ctx.solver = model._solver
		ctx.opt = model._opt
		ctx.theory = model.theory
		ctx.model = model
		ctx.is_default_mask = is_default_mask
		ctx.grad_scaling = grad_scaling

		y, x_flip = Z3SolverOp._assert_and_check(x, mask, ctx.z3_vars, None, ctx.solver, ctx.opt, 
														do_maxsat=do_maxsat_forward,
														return_cores=False, is_default_mask=is_default_mask)
		
		ctx.save_for_backward(x, mask, y, x_flip)

		return y

	@staticmethod
	def backward(
		ctx: torch.autograd.function._ContextMethodMixin, 
		grad_output: torch.Tensor
	) -> tuple[torch.Tensor, None, torch.Tensor, None, None, None]:
		"""
			Backward pass of the layer. Takes the gradient of the loss
			with respect to the layer's output, and returns the gradient
			of the loss with respect to the layer's input.

			Arguments:
				ctx:            Context object
				grad_output:    Gradient of the loss with respect to the
								layer's output
			
			Returns: The gradient of the loss with respect to the layer's
				input.
		"""
		
		# Recover state saved in the forward pass
		x, mask, y, x_flip = ctx.saved_tensors
		z3_vars = ctx.z3_vars
		var_ids, clause_ids = ctx.var_ids, ctx.clause_ids
		solver = ctx.solver
		model = ctx.model
		theory = ctx.theory
		is_default_mask = ctx.is_default_mask
		grad_scaling = ctx.grad_scaling
		
		in_device = x.device
		
		grad_x = grad_mask = grad_model = grad_is_default_mask = None

		# The basic idea is to use the provided gradient
		# to identify what the solver *should have* computed
		# in the forward pass.
		# We compute this by adding the provided gradient
		# direction to the solver's previous answer.
		grad_output_sign = torch.sign(grad_output)
		new_y = torch.sign(torch.sign(y) - 2*grad_output_sign)

		x *= x_flip
		x_sign = torch.sign(x)
		grad_x = torch.zeros_like(x)

		bce = torch.nn.BCEWithLogitsLoss()
		x_clone = x.clone().detach().requires_grad_(True)
		with torch.enable_grad():
			loss = bce(x_clone, torch.relu(x_sign))
		loss.backward()
		grad_plus = x_clone.grad.clone().detach()

		x_clone = x.clone().detach().requires_grad_(True)
		with torch.enable_grad():
			loss = bce(x_clone, 1.0 - torch.relu(x_sign))
		loss.backward()
		grad_minus = x_clone.grad.clone().detach()

		if is_default_mask:
			z3_inputs, z3_outputs, input_ids = Z3SolverOp._get_input_output_vars(mask[0], z3_vars, var_ids)
			n_outputs = len(z3_outputs)
		
		for i, new_yi in enumerate(new_y):

			pad_mask = new_y[i] != 2
			new_yi = new_y[i][pad_mask]
			yi = y[i][pad_mask]

			if not is_default_mask:
				z3_inputs, z3_outputs, input_ids = Z3SolverOp._get_input_output_vars(mask[i], z3_vars, var_ids)
				n_outputs = len(z3_outputs)

			if torch.all(new_yi == yi).item():
				grad_x[i] += grad_plus[i]
				continue

			# construct assertions for the "corrected" solver outputs
			new_y_cons = Z3SolverOp._z3_var_constraints(new_yi, z3_outputs)

			solver.push()

			solver.add(new_y_cons)

			li = Z3SolverOp._floatvec_to_cons(x[i][mask[i]==1], z3_inputs)
			for j, xcon in enumerate(li):
				if model._is_fixed[z3_inputs[j]]:
					solver.add(xcon)
				else:
					solver.assert_and_track(xcon, input_ids[j])

			# check satisfiability of the current solver state
			if solver.check() == z3.sat:
				is_sat = True
			else:
				core = solver.unsat_core()
				is_sat = False

			solver.pop()

			if not is_sat:
				indices = [int(str(phi)[3:]) for phi in core if str(phi).startswith('var')]
				if grad_scaling is not None:
					sws = F.softmax(-grad_scaling*torch.abs(x[i,indices]), dim=-1)
				
				for j, idx in enumerate(indices):
					if grad_scaling is None:
						grad_x[i, idx] += grad_minus[i, idx]
					else:
						grad_x[i, idx] += sws[j]*grad_minus[i, idx]
		
		del x_clone

		grad_x *= x_flip

		return grad_x, None, grad_mask, None, grad_model, grad_is_default_mask

class Z3SolverOp_MaxSat(Z3SolverOp):
	"""
		Variant of `Z3SolverOp` that uses the MaxSAT solver in its backward pass.
	"""

	@staticmethod
	def backward(
		ctx: torch.autograd.function._ContextMethodMixin,
		grad_output: torch.Tensor
	) -> tuple[torch.Tensor, None, torch.Tensor, None, None, None]:
		"""
			Backward pass of the layer. Takes the gradient of the loss
			with respect to the layer's output, and returns the gradient
			of the loss with respect to the layer's input. Uses the
			MaxSAT approach to approximate the gradient.

			Arguments:
				ctx:            Context object
				grad_output:    Gradient of the loss with respect to the
								layer's output
			
			Returns: The gradient of the loss with respect to the layer's
				input.
		"""
		
		# Recover state saved in the forward pass
		x, mask, y, x_flip = ctx.saved_tensors
		z3_vars = ctx.z3_vars
		var_ids, clause_ids = ctx.var_ids, ctx.clause_ids
		solver = ctx.solver
		opt = ctx.model._opt
		model = ctx.model
		theory = ctx.theory
		is_default_mask = ctx.is_default_mask
		grad_scaling = ctx.grad_scaling
		
		in_device = x.device
		
		grad_x = grad_mask = grad_model = grad_is_default_mask = None

		# The basic idea is to use the provided gradient
		# to identify what the solver *should have* computed
		# in the forward pass.
		# We compute this by adding the provided gradient
		# direction to the solver's previous answer.
		grad_output_sign = torch.sign(grad_output)
		new_y = torch.sign(torch.sign(y) - 2*grad_output_sign)

		x *= x_flip
		x_sign = torch.sign(x)
		grad_x = torch.zeros_like(x)

		bce = torch.nn.BCEWithLogitsLoss()
		x_clone = x.clone().detach().requires_grad_(True)
		with torch.enable_grad():
			loss = bce(x_clone, torch.relu(x_sign))
		loss.backward()
		grad_plus = x_clone.grad.clone().detach()

		x_clone = x.clone().detach().requires_grad_(True)
		with torch.enable_grad():
			loss = bce(x_clone, 1.0 - torch.relu(x_sign))
		loss.backward()
		grad_minus = x_clone.grad.clone().detach()

		if is_default_mask:
			z3_inputs, z3_outputs, input_ids = Z3SolverOp._get_input_output_vars(mask[0], z3_vars, var_ids)
			n_outputs = len(z3_outputs)
		
		for i, new_yi in enumerate(new_y):

			weight = torch.abs(grad_output[i]).sum()

			if torch.all(new_yi == y[i]).item():
				grad_x[i] += grad_plus[i]
				continue

			if not is_default_mask:
				z3_inputs, z3_outputs, input_ids = Z3SolverOp._get_input_output_vars(mask[i], z3_vars, var_ids)
				n_outputs = len(z3_outputs)

			# construct assertions for the "corrected" solver outputs
			new_y_cons = Z3SolverOp._z3_var_constraints(new_yi, z3_outputs)

			opt.push()
			opt.add(new_y_cons)
			li = Z3SolverOp._floatvec_to_cons(x[i][mask[i]==1], z3_inputs)
			probits = F.softmax(torch.abs(x[i][mask[i]==1]), dim=-1)
			for j, xcon in enumerate(li):
				if model._is_fixed[z3_inputs[j]]:
					opt.add(xcon)
				else:
					opt.add_soft(xcon, probits[j].item())

			for k in range(1):
				opt.check()
				try:
					asgn = opt.model()
				except:
					break
				m = lambda x: asgn.evaluate(x, model_completion=True)

				indices_minus = [int(str(input_ids[j])[3:]) for j, xcon in enumerate(li) if z3.is_false(m(xcon))]

				if grad_scaling is None:
					grad_x[i, indices_minus] += grad_minus[i, indices_minus]
				else:
					sws = F.softmax(-grad_scaling*torch.abs(x[i,indices_minus]), dim=-1)
					grad_x[i, indices_minus] += sws*grad_minus[i, indices_minus]

				opt.add(z3.Not(z3.And([xcon == m(xcon) for xcon in li])))

			opt.pop()
		
		del x_clone

		grad_x *= x_flip

		return grad_x, None, grad_mask, None, grad_model, grad_is_default_mask

class SMTLayer(torch.nn.Module):
	"""
		A PyTorch module that implements a layer that uses an SMT solver
		to compute the output of a neural network layer. The layer is
		implemented as a torch.autograd.Function, and can be used in
		both training and inference modes.
	"""

	def __init__(
		self,
		input_size: int,
		output_size: int,
		theory: list[z3.BoolRef],
		variables: list[z3.BoolRef]=None,
		default_mask: torch.Tensor=None,
		fixed_inputs: list[z3.BoolRef]=None,
		solverop: str='smt'
	):
		
		super().__init__()
		
		self.input_size = input_size
		self.output_size = output_size
		self.theory = theory
		self.default_mask = default_mask
		self.fixed_inputs = fixed_inputs
		self._n_vars = input_size + output_size

		self._is_fixed = {v: False for v in variables}
		if fixed_inputs is not None:
			self._is_fixed.update({v: True for v in fixed_inputs})

		if variables is None:
			self._z3_vars = [z3.Bool('v{}'.format(i)) for i in range(self._n_vars)]
		else:
			self._n_vars = len(variables)
			self._z3_vars = variables
				
		self._var_ids = [z3.Bool('var{}'.format(i)) for i in range(len(self._z3_vars))]
		self._clause_ids = [z3.Bool('clause{}'.format(i)) for i in range(len(theory))]

		self._solver = z3.Solver()
		self._opt = z3.Optimize()
		self._solverop = Z3SolverOp if solverop == 'smt' else Z3SolverOp_MaxSat
		self._solver.set(':core.minimize', True)

		self._solverop._assert_clauses(theory, None, self._solver)
		self._solverop._assert_clauses(theory, None, self._opt)
				
	def forward(
		self, 
		x: torch.Tensor, 
		mask: torch.Tensor=None,
		grad_scaling: float=None,
		do_maxsat_forward: bool=False
	) -> torch.Tensor:
		"""
			Forward pass of the layer. Takes a batch of floating-point inputs,
			and returns a batch of floating-point outputs.

			Arguments:
				x:          Floating-point input
				mask:       Mask of input and output variables
				grad_scaling: Scaling factor for the gradient
				do_maxsat_forward: Whether to use maxsat. Defaults to false, but if
									true, then the maxsat engine is used instead of sat
			
			Returns: A batch of floating-point outputs.
		"""

		if mask is not None and mask.shape != x.shape:
			raise ValueError('mask and input shapes do not match')

		if mask is None and self.default_mask is None:
			raise ValueError('no mask provided and default mask is undefined')

		if mask is None:
			mask = self.default_mask.unsqueeze(0).repeat(len(x), 1)
			is_default_mask = True
		else:
			is_default_mask = False
		
		return self._solverop.apply(
			x,
			do_maxsat_forward,
			mask,
			grad_scaling,
			self,
			is_default_mask)

