import copy
import functools
import typing as T

import circuits as cc
import gym
import gym.spaces
import gym.spaces.utils
import numpy as np
import torch
import utils.torch as utt
from tonic.torch import models, normalizers

from . import unflat


class RhythmGenerationOscKwargs(T.TypedDict, total=False):
  adaptation_time: float
  active_time: float
  quiet_time: float
  active_scale: float
  quiet_scale: float
  tonic_threshold: float


class RhythmGenerationBasicKwargs(T.TypedDict, total=False):
  voltage_time: float


def init_rg_default(p: utt.containers.ParameterManager, grad: bool = False):
  # Unit biases.
  p.config('bias_flx', ['bias_flx_*'], 0.1, grad)
  p.config('bias_ext', ['bias_ext_*'], 0.8, grad)
  p.config('bias_v0d', ['bias_v0d_*'], -0.1, grad)
  p.config('bias_v0v', ['bias_v0v_*'], -0.1, grad)
  p.config('bias_v3f', ['bias_v3f_*'], -0.4, grad)
  p.config('bias_v3e', ['bias_v3e_*'], -0.0, grad)
  p.config('bias_v3a', ['bias_v3a_*'], -0.4, grad)
  p.config('bias_in2', ['bias_in2_*'], -0.1, grad)
  # Oscillator connections (osc).
  p.config('osc_flxext_i', ['osc_flxext_i_*'], -1.0, grad)
  p.config('osc_extflx_i', ['osc_extflx_i_*'], -0.1, grad)
  # Cross-side connections (cross).
  p.config('cross_flxflx_i', ['cross_flxflx_i_*'], -1.5, grad)
  p.config('cross_flxflx_e', ['cross_flxflx_e_*'], 0.3, grad)
  p.config('cross_extext_e', ['cross_extext_e_*'], 0.05, grad)
  # Same-side, descending connections (sided).
  p.config('sided_flxflx_i', ['sided_flxflx_i_*'], -0.1, grad)
  p.config('sided_extflx_e', ['sided_extflx_e_*'], 0.1, grad)
  # Same-side, ascending connections (sidea).
  p.config('sidea_extflx_e', ['sidea_extflx_e_*'], 0.1, grad)
  # Diagonal, descending connections (diagd).
  p.config('diagd_flxflx_i', ['diagd_flxflx_i_*'], -0.8, grad)
  p.config('diagd_flxflx_e', ['diagd_flxflx_e_*'], 0.2, grad)
  p.config('diagd_in2v0d_i', ['diagd_in2v0d_i_*'], -0.2, grad)
  # Diagonal, ascending connections (diaga).
  p.config('diaga_flxflx_e', ['diaga_flxflx_e_*'], 0.2, grad)
  p.config('diaga_v3ain2_e', ['diaga_v3ain2_e_*'], 0.8, grad)
  # Afferent connections (aff).
  p.config('aff_flx', ['aff_flx_*'], 1.0, grad)
  p.config('aff_ext', ['aff_ext_*'], 1.0, grad)
  # Command connections (freq, sync).
  p.config('freq_osc_e', ['freq_osc_e_*'], 1.0, grad)
  p.config('freq_cross_e', ['freq_cross_e_*'], 1.0, grad)
  p.config('freq_diaga_e', ['freq_diaga_e_*'], 0.8, grad)
  p.config('sync_cross_i', ['sync_cross_i_*'], -0.8, grad)
  p.config('sync_diagdi_i', ['sync_diagdi_i_*'], -0.5, grad)
  p.config('sync_diagde_i', ['sync_diagde_i_*'], -0.5, grad)


def init_rg_cross(p: utt.containers.ParameterManager, grad: bool = False):
  # Unit biases.
  p.config('bias_flx', ['bias_flx_*'], 0.5, grad)  # tonic ext keeps flx at -0.5 + 0.5 = 0.0
  p.config('bias_ext', ['bias_ext_*'], 1.0, grad)  # tonic ext
  p.config('bias_v0d', ['bias_v0d_*'], -0.1, grad)
  p.config('bias_v0v', ['bias_v0v_*'], -0.1, grad)
  p.config('bias_v3f', ['bias_v3f_*'], -0.5, grad)
  p.config('bias_v3e', ['bias_v3e_*'], -0.5, grad)
  p.config('bias_v3a', ['bias_v3a_*'], -0.5, grad)
  p.config('bias_in2', ['bias_in2_*'], -0.5, grad)
  # Oscillator connections (osc).
  p.config('osc_flxext_i', ['osc_flxext_i_*'], -1.5, grad)  # flx dominates
  p.config('osc_extflx_i', ['osc_extflx_i_*'], -0.5, grad)  # ext has minor influence
  # Cross-side connections (cross).
  p.config('cross_flxflx_i', ['cross_flxflx_i_*'], -1.0, grad)
  p.config('cross_flxflx_e', ['cross_flxflx_e_*'], 0.1, grad)
  p.config('cross_extext_e', ['cross_extext_e_*'], 0.1, grad)
  # Afferent connections (aff).
  p.config('aff_flx', ['aff_flx_*'], 1.0, False)
  p.config('aff_ext', ['aff_ext_*'], 1.0, False)
  # Command connections (freq, sync).
  p.config('freq_osc_e', ['freq_osc_e_*'], 1.0, grad)
  p.config('freq_cross_e', ['freq_cross_e_*'], 0.5, grad)
  p.config('sync_cross_i', ['sync_cross_i_*'], -0.5, grad)


def init_rg_full(p: utt.containers.ParameterManager, grad: bool = False):
  # Unit biases.
  p.config('bias_flx', ['bias_flx_*'], 0.7, grad)  # tonic ext keeps flx at -0.5 + 0.7 = 0.2
  p.config('bias_ext', ['bias_ext_*'], 1.0, grad)  # tonic ext
  p.config('bias_v0d', ['bias_v0d_*'], -0.1, grad)
  p.config('bias_v0v', ['bias_v0v_*'], -0.1, grad)
  p.config('bias_v3e', ['bias_v3e_*'], -0.5, grad)  # inactive at slow
  p.config('bias_v3f', ['bias_v3f_*'], -0.5, grad)  # inactive at slow
  p.config('bias_v3a', ['bias_v3a_*'], -0.5, grad)  # inactive at slow
  p.config('bias_in2', ['bias_in2_*'], -0.5, grad)
  # Oscillator connections (osc).
  p.config('osc_flxext_i', ['osc_flxext_i_*'], -1.5, grad)  # flx dominates
  p.config('osc_extflx_i', ['osc_extflx_i_*'], -0.5, grad)  # ext has minor influence
  # Cross-side connections (cross).
  p.config('cross_flxflx1_i', ['cross_flxflx_i_*L'], -1.7, grad)  # always async
  p.config('cross_flxflx2_i', ['cross_flxflx_i_*R'], -1.5, grad)  # always async
  p.config('cross_flxflx_e', ['cross_flxflx_e_*'], 0.4, grad)  #
  p.config('cross_extext_e', ['cross_extext_e_*'], 0.1, grad)  #
  # Same-side, descending connections (sided).
  p.config('sided_flxflx_i', ['sided_flxflx_i_*'], -0.5, grad)
  p.config('sided_extflx_e', ['sided_extflx_e_*'], 0.1, grad)
  # Same-side, ascending connections (sidea).
  p.config('sidea_extflx_e', ['sidea_extflx_e_*'], 0.1, grad)
  # Diagonal, descending connections (diagd).
  p.config('diagd_flxflx_i', ['diagd_flxflx_i_*'], -1.5, grad)  # diag async at slow
  p.config('diagd_flxflx_e', ['diagd_flxflx_e_*'], 0.8, grad)  # diag sync at fast
  p.config('diagd_in2v0d_i', ['diagd_in2v0d_i_*'], -0.1, grad)
  # Diagonal, ascending connections (diaga).
  p.config('diaga_flxflx_e', ['diaga_flxflx_e_*'], 0.8, grad)
  p.config('diaga_v3ain2_e', ['diaga_v3ain2_e_*'], 0.1, grad)
  # Afferent connections (aff).
  p.config('aff_flx', ['aff_flx_*'], 1.0, False)  #
  p.config('aff_ext', ['aff_ext_*'], 1.0, False)  #
  # Command connections (freq, sync).
  p.config('freq_osc_e', ['freq_osc_e_*'], 1, grad)
  p.config('freq_cross_e', ['freq_cross_e_*'], 1.2, grad)
  p.config('freq_diaga_e', ['freq_diaga_e_*'], 0.5, grad)
  p.config('sync_cross_i', ['sync_cross_i_*'], -0.8, grad)  # always async
  p.config('sync_diagdi_i', ['sync_diagdi_i_*'], -1.0, grad)  # diag async at slow
  p.config('sync_diagde_i', ['sync_diagde_i_*'], -0.1, grad)  # diag sync at fast


class RhythmGenerationCircuit(cc.Group):
  def __init__(
    self,
    init: T.Callable[[utt.containers.ParameterManager], None] = init_rg_default,
    osc_kwargs: RhythmGenerationOscKwargs = {},
    basic_kwargs: RhythmGenerationBasicKwargs = {},
    **kwargs,
  ):
    super().__init__(**kwargs)

    # ----------------------------------------------------------------------------------------------
    # Parameters

    self.params = p = utt.containers.ParameterManager()
    p.config('zero', ['*'], 0., False)
    init(p)

    # ----------------------------------------------------------------------------------------------
    # Circuit units

    osc_kwargs = {
      'adaptation_time': 1400,
      'active_time': 200,
      'quiet_time': 1200,
      'active_scale': 100 / 200,
      'quiet_scale': 60 / 1200,
      'tonic_threshold': 3,
      **osc_kwargs,
    }
    basic_kwargs = {
      'voltage_time': 30,
      **basic_kwargs,
    }

    # Oscillator units.
    XF = -20
    XH = -70
    XFLX = 0
    YFLX = 20
    XEXT = 0
    YEXT = 23
    # FL
    self.flx_FL = cc.Oscillator(
      bias=p['bias_flx_FL'], **osc_kwargs, xy=(XF - XFLX, +YFLX), anchor='f'
    )
    self.ext_FL = cc.Basic(bias=p['bias_ext_FL'], **basic_kwargs, xy=(XF - XEXT, +YEXT))
    # FR
    self.flx_FR = cc.Oscillator(
      bias=p['bias_flx_FR'], **osc_kwargs, xy=(XF - XFLX, -YFLX), anchor='f'
    )
    self.ext_FR = cc.Basic(bias=p['bias_ext_FR'], **basic_kwargs, xy=(XF - XEXT, -YEXT))
    # HL
    self.flx_HL = cc.Oscillator(
      bias=p['bias_flx_HL'], **osc_kwargs, xy=(XH - XFLX, +YFLX), anchor='f'
    )
    self.ext_HL = cc.Basic(bias=p['bias_ext_HL'], **basic_kwargs, xy=(XH - XEXT, +YEXT))
    # HR
    self.flx_HR = cc.Oscillator(
      bias=p['bias_flx_HR'], **osc_kwargs, xy=(XH - XFLX, -YFLX), anchor='f'
    )
    self.ext_HR = cc.Basic(bias=p['bias_ext_HR'], **basic_kwargs, xy=(XH - XEXT, -YEXT))

    # Interneuron units.
    XC_V0D = XFLX
    XC_V3F = XFLX - 5
    XC_V3E = XEXT - 5
    YC_V0D = YFLX - 13
    YC_V3F = YFLX - 11
    YC_V3E = YEXT + 4
    XD_V0D = 12
    XD_V0V = 12
    XD_IN2 = 6
    YD_V0D = 7
    YD_V0V = 11
    YD_IN2 = 5
    XA_V3A = -12
    YA_V3A = 11
    # FL
    self.cross_v0d_FL = cc.Basic(
      bias=p['bias_v0d_FL'], **basic_kwargs, xy=(XF - XC_V0D, +YC_V0D), flow='f', anchor='f'
    )
    self.cross_v3f_FL = cc.Basic(
      bias=p['bias_v3f_FL'], **basic_kwargs, xy=(XF - XC_V3F, +YC_V3F), flow='f', anchor='f'
    )
    self.cross_v3e_FL = cc.Basic(
      bias=p['bias_v3e_FL'], **basic_kwargs, xy=(XF - XC_V3E, +YC_V3E), flow='f'
    )
    self.diagd_v0d_FL = cc.Basic(
      bias=p['bias_v0d_FL'], **basic_kwargs, xy=(XF - XD_V0D, +YD_V0D), anchor='f'
    )
    self.diagd_v0v_FL = cc.Basic(
      bias=p['bias_v0v_FL'], **basic_kwargs, xy=(XF - XD_V0V, +YD_V0V), anchor='f'
    )
    self.diagd_in2_FL = cc.Basic(
      bias=p['bias_in2_FL'], **basic_kwargs, xy=(XF - XD_IN2, +YD_IN2), anchor='f'
    )
    # FR
    self.cross_v0d_FR = cc.Basic(
      bias=p['bias_v0d_FR'], **basic_kwargs, xy=(XF - XC_V0D, -YC_V0D), flow='f', anchor='f'
    )
    self.cross_v3f_FR = cc.Basic(
      bias=p['bias_v3f_FR'], **basic_kwargs, xy=(XF - XC_V3F, -YC_V3F), flow='f', anchor='f'
    )
    self.cross_v3e_FR = cc.Basic(
      bias=p['bias_v3e_FR'], **basic_kwargs, xy=(XF - XC_V3E, -YC_V3E), flow='f'
    )
    self.diagd_v0d_FR = cc.Basic(
      bias=p['bias_v0d_FR'], **basic_kwargs, xy=(XF - XD_V0D, -YD_V0D), anchor='f'
    )
    self.diagd_v0v_FR = cc.Basic(
      bias=p['bias_v0v_FR'], **basic_kwargs, xy=(XF - XD_V0V, -YD_V0V), anchor='f'
    )
    self.diagd_in2_FR = cc.Basic(
      bias=p['bias_in2_FR'], **basic_kwargs, xy=(XF - XD_IN2, -YD_IN2), anchor='f'
    )
    # HL
    self.cross_v0d_HL = cc.Basic(
      bias=p['bias_v0d_HL'], **basic_kwargs, xy=(XH - XC_V0D, +YC_V0D), flow='f', anchor='f'
    )
    self.cross_v3f_HL = cc.Basic(
      bias=p['bias_v3f_HL'], **basic_kwargs, xy=(XH - XC_V3F, +YC_V3F), flow='f', anchor='f'
    )
    self.cross_v3e_HL = cc.Basic(
      bias=p['bias_v3e_HL'], **basic_kwargs, xy=(XH - XC_V3E, +YC_V3E), flow='f'
    )
    self.diaga_v3a_HL = cc.Basic(
      bias=p['bias_v3a_HL'], **basic_kwargs, xy=(XH - XA_V3A, +YA_V3A), flow='f', anchor='f'
    )
    # HR
    self.cross_v0d_HR = cc.Basic(
      bias=p['bias_v0d_HR'], **basic_kwargs, xy=(XH - XC_V0D, -YC_V0D), flow='f', anchor='f'
    )
    self.cross_v3f_HR = cc.Basic(
      bias=p['bias_v3f_HR'], **basic_kwargs, xy=(XH - XC_V3F, -YC_V3F), flow='f', anchor='f'
    )
    self.cross_v3e_HR = cc.Basic(
      bias=p['bias_v3e_HR'], **basic_kwargs, xy=(XH - XC_V3E, -YC_V3E), flow='f'
    )
    self.diaga_v3a_HR = cc.Basic(
      bias=p['bias_v3a_HR'], **basic_kwargs, xy=(XH - XA_V3A, -YA_V3A), flow='f', anchor='f'
    )

    # Afferent units.
    XAF_FLX = XFLX - 5
    YAF_FLX = YFLX - 5
    YAF_FLXH = YFLX - 7
    XAF_EXT = XEXT - 5
    YAF_EXT = YEXT + 2
    # FL
    self.aff_flx_FL = cc.Signal(xy=(XF - XAF_FLX, +YAF_FLX), anchor='f')
    self.aff_ext_FL = cc.Signal(xy=(XF - XAF_EXT, +YAF_EXT))
    # FR
    self.aff_flx_FR = cc.Signal(xy=(XF - XAF_FLX, -YAF_FLX), anchor='f')
    self.aff_ext_FR = cc.Signal(xy=(XF - XAF_EXT, -YAF_EXT))
    # HL
    self.aff_flx_HL = cc.Signal(xy=(XH - XAF_FLX, +YAF_FLXH), anchor='f')
    self.aff_ext_HL = cc.Signal(xy=(XH - XAF_EXT, +YAF_EXT))
    # HR
    self.aff_flx_HR = cc.Signal(xy=(XH - XAF_FLX, -YAF_FLXH), anchor='f')
    self.aff_ext_HR = cc.Signal(xy=(XH - XAF_EXT, -YAF_EXT))

    # Command units.
    XCMD = -1
    YFREQ = 3
    YSYNC = 1
    # R
    self.freq_R = cc.Signal(xy=(XCMD, -YFREQ), anchor='f')
    self.sync_R = cc.Signal(xy=(XCMD, -YSYNC))
    # L
    self.freq_L = cc.Signal(xy=(XCMD, +YFREQ), anchor='f')
    self.sync_L = cc.Signal(xy=(XCMD, +YSYNC))

    # ----------------------------------------------------------------------------------------------
    # Circuit connections

    # Alias self to ensure that named access of circuit units typechecks correctly.
    u = T.cast(T.Mapping[str, cc.Basic | cc.Oscillator], self)

    # Oscillator connections (osc).
    for a in ('FL', 'FR', 'HL', 'HR'):
      u[f'flx_{a}'].synapse(u[f'ext_{a}'], weight=p[f'osc_extflx_i_{a}'], sign='-')
      u[f'ext_{a}'].synapse(u[f'flx_{a}'], weight=p[f'osc_flxext_i_{a}'], sign='-')

    # Cross-side connections (cross).
    for a, b in (('FR', 'FL'), ('FL', 'FR'), ('HR', 'HL'), ('HL', 'HR')):
      u[f'cross_v3f_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'cross_v0d_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'cross_v3e_{a}'].synapse(u[f'ext_{a}'], weight=1, sign='+')
      u[f'flx_{b}'].synapse(u[f'cross_v0d_{a}'], weight=p[f'cross_flxflx_i_{b}'], sign='-')
      u[f'flx_{b}'].synapse(u[f'cross_v3f_{a}'], weight=p[f'cross_flxflx_e_{b}'], sign='+')
      u[f'ext_{b}'].synapse(
        u[f'cross_v3e_{a}'], weight=p[f'cross_extext_e_{b}'], sign='+', waypoints=[(-8, -1, 'y-')]
      )

    # Same-side, descending connections (sided).
    for a, b in (('FL', 'HL'), ('FR', 'HR')):
      u[f'flx_{b}'].synapse(
        u[f'ext_{a}'], weight=p[f'sided_extflx_e_{b}'], sign='+', waypoints=[(-10, -2, 'y-')]
      )
      u[f'flx_{b}'].synapse(
        u[f'flx_{a}'], weight=p[f'sided_flxflx_i_{b}'], sign='-', waypoints=[(-11, +2, 'y+')]
      )

    # Same-side, ascending connections (sidea).
    for a, b in (('HL', 'FL'), ('HR', 'FR')):
      u[f'flx_{b}'].synapse(
        u[f'ext_{a}'], weight=p[f'sidea_extflx_e_{b}'], sign='+', waypoints=[(-3, -7, 'y-')]
      )

    # Diagonal, descending connections (diagd).
    for a, b in (('FL', 'HR'), ('FR', 'HL')):
      u[f'diagd_v0d_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'diagd_v0v_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'flx_{b}'].synapse(
        u[f'diagd_v0d_{a}'],
        weight=p[f'diagd_flxflx_i_{b}'],
        sign='-',
        waypoints=[(-35, +12, 'y-')]
      )
      u[f'flx_{b}'].synapse(
        u[f'diagd_v0v_{a}'],
        weight=p[f'diagd_flxflx_e_{b}'],
        sign='+',
        waypoints=[(-34, +12, 'y-')]
      )
      u[f'diagd_in2_{a}'].synapse(u[f'cross_v3f_{a}'], weight=1, sign='+')
      u[f'diagd_v0d_{a}'].synapse(u[f'diagd_in2_{a}'], weight=p[f'diagd_in2v0d_i_{a}'], sign='-')

    # Diagonal, ascending connections (diaga).
    for a, b in (('HL', 'FR'), ('HR', 'FL')):
      u[f'diaga_v3a_{a}'].synapse(u[f'flx_{a}'], weight=1, sign='+')
      u[f'flx_{b}'].synapse(
        u[f'diaga_v3a_{a}'], weight=p[f'diaga_flxflx_e_{b}'], sign='+', waypoints=[(32, 7.5, 'x-')]
      )
      u[f'diagd_in2_{b}'].synapse(
        u[f'diaga_v3a_{a}'],
        weight=p[f'diaga_v3ain2_e_{b}'],
        sign='+',
        waypoints=[(26, -7.5, 'x-')]
      )

    # Afferent connections (aff).
    for a in ('FL', 'FR', 'HL', 'HR'):
      u[f'flx_{a}'].synapse(u[f'aff_flx_{a}'], weight=p[f'aff_flx_{a}'])
      u[f'ext_{a}'].synapse(u[f'aff_ext_{a}'], weight=p[f'aff_ext_{a}'])

    # Command connections (freq, sync).
    for a in ('L', 'R'):
      u[f'flx_F{a}'].synapse(
        u[f'freq_{a}'], weight=p[f'freq_osc_e_{a}'], sign='+', waypoints=[(-10, 12, 'y-')]
      )
      u[f'flx_H{a}'].synapse(
        u[f'freq_{a}'], weight=p[f'freq_osc_e_{a}'], sign='+', waypoints=[(-10, 12, 'y-')]
      )
      u[f'cross_v3f_F{a}'].synapse(u[f'freq_{a}'], weight=p[f'freq_cross_e_{a}'], sign='+')
      u[f'cross_v3f_H{a}'].synapse(u[f'freq_{a}'], weight=p[f'freq_cross_e_{a}'], sign='+')
      u[f'diaga_v3a_H{a}'].synapse(
        u[f'freq_{a}'], weight=p[f'freq_diaga_e_{a}'], sign='+', waypoints=[(-2, 3, 'y-')]
      )
      u[f'cross_v0d_F{a}'].synapse(u[f'sync_{a}'], weight=p[f'sync_cross_i_{a}'], sign='-')
      u[f'cross_v0d_H{a}'].synapse(u[f'sync_{a}'], weight=p[f'sync_cross_i_{a}'], sign='-')
      u[f'diagd_v0d_F{a}'].synapse(
        u[f'sync_{a}'], weight=p[f'sync_diagdi_i_{a}'], sign='-', waypoints=[(-2, 3.5, 'y-')]
      )
      u[f'diagd_v0v_F{a}'].synapse(
        u[f'sync_{a}'], weight=p[f'sync_diagde_i_{a}'], sign='-', waypoints=[(-2, 3, 'y-')]
      )


def constrain_sign(
  param: torch.nn.Parameter | torch.Tensor,
  magnitude: bool = False,
  zero: bool = False,
):
  def _constrain_min(x: float) -> float:
    xmin = x if magnitude else 0
    if zero:
      return xmin if x >= 0 else -float('inf')
    else:
      return xmin if x > 0 else -float('inf')

  def _constrain_max(x: float) -> float:
    xmax = x if magnitude else 0
    if zero:
      return xmax if x <= 0 else float('inf')
    else:
      return xmax if x < 0 else float('inf')

  param.constrain_min = param.data.clone().apply_(_constrain_min)  # type: ignore
  param.constrain_max = param.data.clone().apply_(_constrain_max)  # type: ignore


class Constrainer(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.params: dict[str, torch.nn.Parameter | torch.Tensor] = {}

  def initialize(self, params: T.Mapping[str, torch.nn.Parameter | torch.Tensor]):
    self.params = {
      name: p
      for name,
      p in params.items()
      if hasattr(p, 'constrain_min') or hasattr(p, 'constrain_max')
    }

  @torch.no_grad()
  def constrain(self):
    for p in self.params.values():
      p.data.clamp_(getattr(p, 'constrain_min', None), getattr(p, 'constrain_max', None))

  def extra_repr(self) -> str:
    return f'num={len(self.params)}'


def _init_zeros(module: torch.nn.Module):
  if isinstance(module, torch.nn.Linear):
    torch.nn.init.zeros_(module.weight)
    if module.bias is not None:
      torch.nn.init.zeros_(module.bias)


def init_zeros(module: torch.nn.Module):
  if isinstance(module, torch.nn.Linear):
    torch.nn.init.zeros_(module.weight)
    if module.bias is not None:
      torch.nn.init.zeros_(module.bias)


class LimbSubnetwork(torch.nn.Module):
  def __init__(self, torso: torch.nn.Module | None = None, head: torch.nn.Module | None = None):
    super().__init__()
    self.torso = torso
    self.head = head or models.DeterministicPolicyHead(
      activation=torch.nn.Tanh, bias=True, fn=_init_zeros
    )

  def initialize(self, observation_size, action_size):
    size = observation_size
    if self.torso is not None:
      size = self.torso.initialize(size)  # type: ignore
    self.head.initialize(size, action_size)  # type: ignore

  def forward(self, observation):
    out = observation
    if self.torso is not None:
      out = self.torso(out)
    return self.head(out)


LimbName = T.Literal['FL', 'FR', 'HL', 'HR']


def _split_limbs(
  observations: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  # Observation values in order [FR, FL, RR, RL] (using Unitree API names).
  # Return in order [FL, FR, HL, HR] (using NCAP model names).
  if len(observations) == 0:
    return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
  groups = zip(*[torch.chunk(v, 4, dim=-1) for v in observations.values()])
  FR, FL, HR, HL = [torch.concat(group, dim=-1) for group in groups]
  return FL, FR, HL, HR


def _combine_limbs(
  action_space: gym.spaces.Dict,
  actions: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
  # Action values in order [FL, FR, HL, HR] (using NCAP model names).
  # Return in order [FR, FL, RR, RL] (using Unitree API names).
  sizes = [v.shape[0] // 4 for v in action_space.values()]
  FL, FR, HL, HR = actions
  parts = zip(*[torch.split(group, sizes, dim=-1) for group in (FR, FL, HR, HL)])
  values = [torch.concat(part, dim=-1).squeeze(0) for part in parts]
  return dict(zip(action_space.keys(), values))


PatternFormationMode = T.Literal[
  'separate',
  'share-cross',
  'share-all',
  'merge-cross',
  'merge-side',
  'merge-diag',
  'merge-all',
]


def init_pf_flxext(
  module,
  flx: float | list[float] = -1.6,
  ext: float | list[float] = 1.6,
  repeat: int = 1,
  mode: PatternFormationMode = 'separate',
  constrain: T.Callable[[torch.nn.Parameter | torch.Tensor], None] | bool = False,
):
  if isinstance(module, torch.nn.Linear):
    with torch.no_grad():
      if module.bias is not None:
        module.bias.zero_()
      module.weight.zero_()  # [[o1, o2, ..., flx, ext]]
      blocks = {'merge-cross': 2, 'merge-side': 2, 'merge-diag': 2, 'merge-all': 4}.get(mode, 1)
      rows = module.weight.shape[0] // blocks
      cols = module.weight.shape[1] // blocks
      for i in range(blocks):
        # Diagonal blocks only.
        block = module.weight[i * rows:(i + 1) * rows, i * cols:(i + 1) * cols]
        block[:, -2 * repeat:-repeat] = torch.tensor(flx).reshape(-1, 1) / repeat
        block[:, -repeat:] = torch.tensor(ext).reshape(-1, 1) / repeat
      if constrain == True:
        constrain_sign(module.weight)
      elif constrain:
        constrain(module.weight)


class PatternFormationNetwork(torch.nn.Module):
  def __init__(
    self,
    net: LimbSubnetwork | None = None,
    mode: PatternFormationMode = 'separate',
    ignore: list[str] = [],
    repeat: int = 1,
    rectify: bool = False,
  ):
    super().__init__()
    net = net or LimbSubnetwork(torso=None, head=None)
    self.names: list[str] = []
    self.mode = mode
    self.ignore = ignore
    self.repeat = repeat
    self.rectify = rectify

    match mode:
      case 'separate':
        self.FL = copy.deepcopy(net)
        self.FR = copy.deepcopy(net)
        self.HL = copy.deepcopy(net)
        self.HR = copy.deepcopy(net)
        self.names = ['FL', 'FR', 'HL', 'HR']
      case 'share-cross':
        self.FL = self.FR = copy.deepcopy(net)
        self.HL = self.HR = copy.deepcopy(net)
        self.names = ['FL', 'HL']
      case 'share-all':
        self.FL = self.FR = self.HL = self.HR = copy.deepcopy(net)
        self.names = ['FL']
      case 'merge-cross':
        self.FL_FR = copy.deepcopy(net)
        self.HL_HR = copy.deepcopy(net)
        self.names = ['FL_FR', 'HL_HR']
      case 'merge-side':
        self.FL_HL = copy.deepcopy(net)
        self.FR_HR = copy.deepcopy(net)
        self.names = ['FL_HL', 'FR_HR']
      case 'merge-diag':
        self.FL_HR = copy.deepcopy(net)
        self.FR_HL = copy.deepcopy(net)
        self.names = ['FL_HR', 'FR_HL']
      case 'merge-all':
        self.FL_FR_HL_HR = copy.deepcopy(net)
        self.names = ['FL_FR_HL_HR']
      case _:
        raise ValueError(f'Invalid mode: {mode}')

  def initialize(
    self,
    observation_space: gym.spaces.Dict,
    action_space: gym.spaces.Dict,
    input_size: int = 0,
  ) -> None:
    self.observation_space = observation_space
    self.action_space = action_space
    observation_size = sum(
      obox.shape[1] for oname, obox in observation_space.items() if oname not in self.ignore
    )
    action_size = sum(abox.shape[0] for abox in action_space.values())
    for name in self.names:
      net = T.cast(LimbSubnetwork, self.get_submodule(name))
      match self.mode:
        case 'separate' | 'share-cross' | 'share-all':
          net.initialize(
            observation_size // 4 * (2 if self.rectify else 1) + input_size * self.repeat,
            action_size // 4,
          )
        case 'merge-cross' | 'merge-side' | 'merge-diag':
          net.initialize(
            observation_size // 2 * (2 if self.rectify else 1) + input_size * self.repeat * 2,
            action_size // 2,
          )
        case 'merge-all':
          net.initialize(
            observation_size * (2 if self.rectify else 1) + input_size * self.repeat * 4,
            action_size,
          )

  def forward(
    self,
    observations: dict[str, torch.Tensor],
    inputs: dict[LimbName, torch.Tensor] | None = None,
  ) -> dict[str, torch.Tensor]:
    """
    Args:
      observations: Dict observations with shape (batch_size, observation_value_size).
      inputs: Dict inputs with shape (batch_size, input_size).
    Return:
      actions: Dict actions with shape (batch_size, action_value_size).
    """
    observations = {oname: obox for oname, obox in observations.items() if oname not in self.ignore}
    x_FL, x_FR, x_HL, x_HR = _split_limbs(observations)

    # To convert inputs into biologically-plausible signals, split the input into negative (neg) and
    # positive (pos) components, e.g. [a, b] -> [a-, b-, a+, b+], all non-negative.
    if self.rectify:
      x_FL = torch.concat((torch.clamp(-x_FL, min=0), torch.clamp(x_FL, min=0)), dim=-1)
      x_FR = torch.concat((torch.clamp(-x_FR, min=0), torch.clamp(x_FR, min=0)), dim=-1)
      x_HL = torch.concat((torch.clamp(-x_HL, min=0), torch.clamp(x_HL, min=0)), dim=-1)
      x_HR = torch.concat((torch.clamp(-x_HR, min=0), torch.clamp(x_HR, min=0)), dim=-1)

    if inputs is not None:
      # To leverage the learning properties of high-dimensional spaces, increase the input size
      # with redundant copies, e.g. [flx, ext] with repeat=3 yields [flx, flx, flx, ext, ext, ext].
      x_FL = torch.concat((x_FL, inputs['FL'].repeat_interleave(self.repeat, dim=-1)), dim=-1)
      x_FR = torch.concat((x_FR, inputs['FR'].repeat_interleave(self.repeat, dim=-1)), dim=-1)
      x_HL = torch.concat((x_HL, inputs['HL'].repeat_interleave(self.repeat, dim=-1)), dim=-1)
      x_HR = torch.concat((x_HR, inputs['HR'].repeat_interleave(self.repeat, dim=-1)), dim=-1)

    match self.mode:
      case 'separate' | 'share-cross' | 'share-all':
        y_FL = self.FL(x_FL)
        y_FR = self.FR(x_FR)
        y_HL = self.HL(x_HL)
        y_HR = self.HR(x_HR)
      case 'merge-cross':
        y_FL_FR = self.FL_FR(torch.concat((x_FL, x_FR), dim=-1))
        y_HL_HR = self.HL_HR(torch.concat((x_HL, x_HR), dim=-1))
        y_FL, y_FR = torch.chunk(y_FL_FR, 2, dim=-1)
        y_HL, y_HR = torch.chunk(y_HL_HR, 2, dim=-1)
      case 'merge-side':
        y_FL_HL = self.FL_HL(torch.concat((x_FL, x_HL), dim=-1))
        y_FR_HR = self.FR_HR(torch.concat((x_FR, x_HR), dim=-1))
        y_FL, y_HL = torch.chunk(y_FL_HL, 2, dim=-1)
        y_FR, y_HR = torch.chunk(y_FR_HR, 2, dim=-1)
      case 'merge-diag':
        y_FL_HR = self.FL_HR(torch.concat((x_FL, x_HR), dim=-1))
        y_FR_HL = self.FR_HL(torch.concat((x_FR, x_HL), dim=-1))
        y_FL, y_HR = torch.chunk(y_FL_HR, 2, dim=-1)
        y_FR, y_HL = torch.chunk(y_FR_HL, 2, dim=-1)
      case 'merge-all':
        y_FL_FR_HL_HR = self.FL_FR_HL_HR(torch.concat((x_FL, x_FR, x_HL, x_HR), dim=-1))
        y_FL, y_FR, y_HL, y_HR = torch.chunk(y_FL_FR_HL_HR, 4, dim=-1)

    return _combine_limbs(self.action_space, (y_FL, y_FR, y_HL, y_HR))


AfferentFeedbackMode = T.Literal['separate', 'share-cross', 'share-all', 'merge-cross']

# Joint priors as [hip-, thigh-, calf-, hip+, thigh+, calf+].
# Foot priors as [foot-, foot+].
AF_FLX_IPSI: dict[str, list[float]] = {
  'a1/joints_pos': [0, 0, 0, 0, 0, 0],
  'a1/joints_trq': [0, 0, 0, 0, 0, 0],
  'a1/joints_vel': [0, 0, 0, 0, 0, 0],
  'a1/sensors_foot': [0, 0],
}
AF_EXT_IPSI: dict[str, list[float]] = {
  'a1/joints_pos': [0, 0, 0, 0, 0, 0],
  'a1/joints_trq': [0, 0, 0, 0, 0, 0],
  'a1/joints_vel': [0, 0, 0, 0, 0, 0],
  'a1/sensors_foot': [0, 0],
}
AF_FLX_CONTRA: dict[str, list[float]] = {
  'a1/joints_pos': [0, 0, 0, 0, 0, 0],
  'a1/joints_trq': [0, 0, 0, 0, 0, 0],
  'a1/joints_vel': [0, 0, 0, 0, 0, 0],
  'a1/sensors_foot': [0, 0],
}
AF_EXT_CONTRA: dict[str, list[float]] = {
  'a1/joints_pos': [0, 0, 0, 0, 0, 0],
  'a1/joints_trq': [0, 0, 0, 0, 0, 0],
  'a1/joints_vel': [0, 0, 0, 0, 0, 0],
  'a1/sensors_foot': [0, 0],
}


def _init_af_subblock(
  subblock: torch.Tensor,
  leg_obs_sizes: dict[str, int],
  values: dict[str, list[float]],
):
  repeat = subblock.shape[0]
  cols = subblock.shape[1]
  assert cols == sum(leg_obs_sizes.values()) * 2
  half = cols // 2  # Splits cols into negative and positive halves.
  idx = 0
  for oname, osize in leg_obs_sizes.items():
    if oname in values:
      x = torch.tensor(values[oname]).reshape(1, -1) / repeat
      assert x.shape[1] == 2 * osize
      subblock[:, idx:idx + osize] = x[:, :osize]  # (repeat, osize) <- (1, osize)
      subblock[:, half + idx:half + idx + osize] = x[:, osize:]  # (repeat, osize) <- (1, osize)
    idx += osize


def init_af_flxext(
  net: LimbSubnetwork,
  observation_value_sizes: dict[str, int],
  repeat: int = 1,
  mode: AfferentFeedbackMode = 'separate',
  rectify: bool = False,
  constrain: T.Callable[[torch.nn.Parameter | torch.Tensor], None] | bool = False,
  flx_ipsi: T.Mapping[str, list[float]] = {},
  ext_ipsi: T.Mapping[str, list[float]] = {},
  flx_contra: T.Mapping[str, list[float]] = {},
  ext_contra: T.Mapping[str, list[float]] = {},
):
  # Only initialize biologically-plausible AF net.
  if rectify is False: return
  # Only initialize linear AF net.
  if net.torso is not None: return

  assert isinstance(net.head, models.DeterministicPolicyHead)
  head = T.cast(torch.nn.Linear, net.head.action_layer[0])
  with torch.no_grad():
    if head.bias is not None:
      head.bias.zero_()
    head.weight.zero_()  # [[o1, o2, o3, ...]] or [[o1-, o2-, o3-, ..., o1+, o2+, o3+, ...]]
    blocks = {'merge-cross': 2}.get(mode, 1)
    rows = head.weight.shape[0] // blocks
    cols = head.weight.shape[1] // blocks
    leg_obs_sizes = {oname: osize // 4 for oname, osize in observation_value_sizes.items()}
    for r in range(blocks):
      for c in range(blocks):
        block = head.weight[r * rows:(r + 1) * rows, c * cols:(c + 1) * cols]
        if r == c:
          # Diagonal block (ipsilateral).
          _init_af_subblock(
            block[:repeat, :],
            leg_obs_sizes,
            AF_FLX_IPSI | dict(flx_ipsi),
          )
          _init_af_subblock(
            block[repeat:2 * repeat, :],
            leg_obs_sizes,
            AF_EXT_IPSI | dict(ext_ipsi),
          )
        else:
          # Off-diagonal block (contralateral).
          _init_af_subblock(
            block[:repeat, :],
            leg_obs_sizes,
            AF_FLX_CONTRA | dict(flx_contra),
          )
          _init_af_subblock(
            block[repeat:2 * repeat, :],
            leg_obs_sizes,
            AF_EXT_CONTRA | dict(ext_contra),
          )
    if constrain == True:
      constrain_sign(head.weight)
    elif constrain:
      constrain(head.weight)


class AfferentFeedbackNetwork(torch.nn.Module):
  def __init__(
    self,
    net: LimbSubnetwork | None = None,
    mode: AfferentFeedbackMode = 'separate',
    ignore: list[str] = [],
    repeat: int = 1,
    repeat_agg: T.Literal['mean', 'sum'] = 'sum',
    rectify: bool = False,
    init: T.Callable[..., None] = init_af_flxext,
  ):
    super().__init__()
    net = net or LimbSubnetwork(
      torso=None,
      head=models.DeterministicPolicyHead(
        activation=torch.nn.Identity,  # type: ignore
        bias=True,
        fn=_init_zeros,
      ),
    )
    self.names: list[str] = []
    self.mode = mode
    self.ignore = ignore
    self.repeat = repeat
    self.rectify = rectify
    self.init = init

    match repeat_agg:
      case 'mean':
        self.repeat_agg = functools.partial(torch.mean, dim=-1, keepdim=True)
      case 'sum':
        self.repeat_agg = functools.partial(torch.sum, dim=-1, keepdim=True)
      case _:
        raise ValueError(f'Invalid repeat_agg: {repeat_agg}')

    match mode:
      case 'separate':
        self.FL = copy.deepcopy(net)
        self.FR = copy.deepcopy(net)
        self.HL = copy.deepcopy(net)
        self.HR = copy.deepcopy(net)
        self.names = ['FL', 'FR', 'HL', 'HR']
      case 'share-cross':
        self.FL = self.FR = copy.deepcopy(net)
        self.HL = self.HR = copy.deepcopy(net)
        self.names = ['FL', 'HL']
      case 'share-all':
        self.FL = self.FR = self.HL = self.HR = copy.deepcopy(net)
        self.names = ['FL']
      case 'merge-cross':
        self.FL_FR = copy.deepcopy(net)
        self.HL_HR = copy.deepcopy(net)
        self.names = ['FL_FR', 'HL_HR']
      case _:
        raise ValueError(f'Invalid mode: {mode}')

  def initialize(
    self,
    observation_space: gym.spaces.Dict,
    output_size: int = 2,
  ) -> None:
    self.observation_space = observation_space
    self.output_size = output_size
    observation_value_sizes = {
      oname: obox.shape[1]
      for oname, obox in observation_space.items()
      if oname not in self.ignore
    }
    observation_size = sum(osize for osize in observation_value_sizes.values())
    for name in self.names:
      net = T.cast(LimbSubnetwork, self.get_submodule(name))
      match self.mode:
        case 'separate' | 'share-cross' | 'share-all':
          net.initialize(
            observation_size // 4 * (2 if self.rectify else 1),
            output_size * self.repeat,
          )
        case 'merge-cross':
          net.initialize(
            observation_size // 2 * (2 if self.rectify else 1),
            output_size * self.repeat * 2,
          )
      self.init(
        net, observation_value_sizes, repeat=self.repeat, mode=self.mode, rectify=self.rectify
      )

  def forward(self, observations: dict[str, torch.Tensor]) -> dict[LimbName, torch.Tensor]:
    """
    Args:
      observations: Dict observations with tensor values (batch_size, observation_value_size).
    Return:
      outputs: Dict outputs with tensor values (batch_size, output_size).
    """
    observations = {oname: obox for oname, obox in observations.items() if oname not in self.ignore}
    x_FL, x_FR, x_HL, x_HR = _split_limbs(observations)

    # To convert inputs into biologically-plausible signals, split the input into negative (neg) and
    # positive (pos) components, e.g. [a, b] -> [a_neg, b_neg, a_pos, b_pos], all non-negative.
    if self.rectify:
      x_FL = torch.concat((torch.clamp(-x_FL, min=0), torch.clamp(x_FL, min=0)), dim=-1)
      x_FR = torch.concat((torch.clamp(-x_FR, min=0), torch.clamp(x_FR, min=0)), dim=-1)
      x_HL = torch.concat((torch.clamp(-x_HL, min=0), torch.clamp(x_HL, min=0)), dim=-1)
      x_HR = torch.concat((torch.clamp(-x_HR, min=0), torch.clamp(x_HR, min=0)), dim=-1)

    match self.mode:
      case 'separate' | 'share-cross' | 'share-all':
        y_FL = self.FL(x_FL)
        y_FR = self.FR(x_FR)
        y_HL = self.HL(x_HL)
        y_HR = self.HR(x_HR)
      case 'merge-cross':
        y_FL_FR = self.FL_FR(torch.concat((x_FL, x_FR), dim=-1))
        y_HL_HR = self.HL_HR(torch.concat((x_HL, x_HR), dim=-1))
        y_FL, y_FR = torch.chunk(y_FL_FR, 2, dim=-1)
        y_HL, y_HR = torch.chunk(y_HL_HR, 2, dim=-1)

    # To leverage the learning properties of high-dimensional spaces, decrease the output size by
    # aggregating over redundant copies, e.g. [flx, flx, flx, ext, ext, ext] with repeat=3 yields
    # [flx, ext].
    if self.repeat > 1:
      y_FL = torch.concat([self.repeat_agg(y) for y in y_FL.split(self.repeat, dim=-1)], dim=-1)
      y_FR = torch.concat([self.repeat_agg(y) for y in y_FR.split(self.repeat, dim=-1)], dim=-1)
      y_HL = torch.concat([self.repeat_agg(y) for y in y_HL.split(self.repeat, dim=-1)], dim=-1)
      y_HR = torch.concat([self.repeat_agg(y) for y in y_HR.split(self.repeat, dim=-1)], dim=-1)

    return {'FL': y_FL, 'FR': y_FR, 'HL': y_HL, 'HR': y_HR}


class BrainstemCommandSignal(torch.nn.Module):
  def __init__(
    self,
    command: float | list[float] | np.ndarray | torch.Tensor,
    grad: bool = True,
    repeat: int = 1,
    repeat_agg: T.Literal['mean', 'sum'] = 'mean',
  ):
    super().__init__()
    if isinstance(command, (float, int)):
      command = torch.tensor([float(command)], dtype=torch.float32)  # (command_size=1,)
    else:
      command = torch.as_tensor(command, dtype=torch.float32)  # (command_size,)
    assert command.ndim == 1
    command = command.unsqueeze(1).repeat_interleave(repeat, dim=-1) / repeat
    self.command = torch.nn.Parameter(command, requires_grad=grad)  # (command_size, repeat)
    self.repeat = repeat

    match repeat_agg:
      case 'mean':
        self.repeat_agg = functools.partial(torch.mean, dim=-1, keepdim=True)
      case 'sum':
        self.repeat_agg = functools.partial(torch.sum, dim=-1, keepdim=True)
      case _:
        raise ValueError(f'Invalid repeat_agg: {repeat_agg}')

  def initialize(
    self,
    observation_space: gym.spaces.Dict,
    output_size: int = 4,
  ) -> None:
    assert output_size % self.command.size(0) == 0
    self.output_size = output_size

  def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Args:
      observations: Dict observations with tensor values (batch_size, observation_value_size).
    Return:
      command: Command signal tensor (1, output_size).
    """
    command = self.repeat_agg(self.command).T  # (1, command_size)
    # Use non-interleaving repeat, e.g. [freq, sync] -> [freq, sync, freq, sync] with output_size=4.
    return command.repeat(1, self.output_size // command.size(1))


def init_bc_fixed(
  module: torch.nn.Module,
  weight: float | list = 0.5,
  bias: float | list = -0.05,
  weight_grad: bool = False,
  bias_grad: bool = False,
):
  if isinstance(module, torch.nn.Linear):
    module.weight.data.copy_(torch.tensor(weight))
    module.weight.requires_grad_(weight_grad)
    if module.bias is not None:
      module.bias.data.copy_(torch.tensor(bias))
      module.bias.requires_grad_(bias_grad)


class BrainstemCommandNetwork(torch.nn.Module):
  def __init__(
    self,
    torso: torch.nn.Module | None = None,
    head: torch.nn.Module | None = None,
    observe: list[str] = ['target_vx'],
    repeat: int = 1,
    command_size: int = 1,
  ):
    super().__init__()
    self.torso = torso
    self.head = head or models.DeterministicPolicyHead(
      activation=torch.nn.Identity,  # type: ignore
      bias=True,
      fn=_init_zeros  # type: ignore
    )
    self.observe = observe
    self.repeat = repeat
    self.command_size = command_size

  def initialize(
    self,
    observation_space: gym.spaces.Dict,
    output_size: int = 4,
  ) -> None:
    self.observation_space = observation_space
    assert output_size % self.command_size == 0
    self.output_size = output_size
    observation_size = sum(
      obox.shape[1] for oname, obox in observation_space.items() if oname in self.observe
    )
    size = observation_size * self.repeat
    if self.torso is not None:
      size = self.torso.initialize(size)  # type: ignore
    self.head.initialize(size, self.command_size)  # type: ignore

  def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Args:
      observations: Dict observations with tensor values (batch_size, observation_value_size).
    Return:
      commands: Command signal tensor (batch_size, output_size).
    """
    out = torch.concat([o for oname, o in observations.items() if oname in self.observe], dim=-1)
    out = out.repeat_interleave(self.repeat, dim=-1)
    if self.torso is not None:
      out = self.torso(out)
    command = self.head(out)  # (batch_size, command_size)
    # Use non-interleaving repeat, e.g. [freq, sync] -> [freq, sync, freq, sync] with output_size=4.
    return command.repeat(1, self.output_size // command.size(1))


class QuadrupedModularActor(torch.nn.Module):
  def __init__(
    self,
    pattern_formation: PatternFormationNetwork,
    rhythm_generation: RhythmGenerationCircuit | None = None,
    afferent_feedback: AfferentFeedbackNetwork | None = None,
    brainstem_command: BrainstemCommandSignal | BrainstemCommandNetwork | None = None,
    constrainer: Constrainer | None = None,
    timestep: float = 30.,  # Match to task.control_timestep = 0.03s = 30ms.
  ):
    super().__init__()

    self.pattern_formation = pattern_formation

    if rhythm_generation is not None:
      self.sim = cc.Simulator(timestep=timestep)
      self.sim.rhythm_generation = rhythm_generation
    else:
      self.sim = None

    assert afferent_feedback is None or rhythm_generation is not None
    self.afferent_feedback = afferent_feedback

    assert brainstem_command is None or rhythm_generation is not None
    self.brainstem_command = brainstem_command

    self.constrainer = constrainer

  def initialize(
    self,
    observation_space: gym.spaces.Dict,
    action_space: gym.spaces.Dict,
    observation_normalizer: unflat.UnflatMeanStd | None = None,
  ):
    # Use Dict spaces, which have Box values in order [FR, FL, RR, RL] (using Unitree API names).
    assert isinstance(observation_space, gym.spaces.Dict)
    assert isinstance(action_space, gym.spaces.Dict)
    assert all(
      isinstance(o, gym.spaces.Box) and o.shape is not None for o in observation_space.values()
    )
    assert all(
      isinstance(a, gym.spaces.Box) and a.shape is not None and a.shape[0] % 4 == 0
      for a in action_space.values()
    )

    if self.sim is not None:
      self.sim.init()
      self.pattern_formation.initialize(observation_space, action_space, 2)  # [flx, ext]
      if self.afferent_feedback is not None:
        self.afferent_feedback.initialize(observation_space, 2)  # [flx, ext]
      if self.brainstem_command is not None:
        self.brainstem_command.initialize(observation_space, 4)  # [freq_L, sync_L, freq_R, sync_R]
    else:
      self.pattern_formation.initialize(observation_space, action_space, 0)

    if self.constrainer:
      self.constrainer.initialize(dict(self.named_parameters()))
      self.constrainer.constrain()

    self.observation_normalizer = observation_normalizer
    if self.observation_normalizer:
      self.observation_normalizer.initialize(observation_space)

  def reset(self):
    if self.sim is not None:
      self.sim.reset()

  def forward(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Args:
      observations: Dict observations with tensor values (batch_size, observation_size).
    Returns:
      actions: Dict actions with tensor values (batch_size, action_size).
    """
    # Normalize observations.
    if self.observation_normalizer:
      observations = self.observation_normalizer(observations)

    inputs: dict[LimbName, torch.Tensor] | None = None
    if self.sim is not None:
      rg = T.cast(RhythmGenerationCircuit, self.sim.rhythm_generation)

      if self.afferent_feedback is not None:
        # Write afferent signals into sim. Assume shape (batch_size=1, 2).
        afferents: dict[LimbName, torch.Tensor] = self.afferent_feedback(observations)
        for limb in ('FL', 'FR', 'HL', 'HR'):
          T.cast(cc.Signal, rg[f'aff_flx_{limb}']).value = afferents[limb][0, 0]
          T.cast(cc.Signal, rg[f'aff_ext_{limb}']).value = afferents[limb][0, 1]

      if self.brainstem_command is not None:
        # Write command signals into sim. Assume shape (batch_size=1, 4).
        commands = self.brainstem_command(observations)
        T.cast(cc.Signal, rg.freq_L).value = commands[0, 0]
        T.cast(cc.Signal, rg.sync_L).value = commands[0, 1]
        T.cast(cc.Signal, rg.freq_R).value = commands[0, 2]
        T.cast(cc.Signal, rg.sync_R).value = commands[0, 3]

      # Any signals written on this timestep will be incorporated at the next timestep.
      self.sim.step()

      # Read oscillator signals from sim.
      inputs = {
        limb:
          torch.concat(
            (
              T.cast(cc.Signal, rg[f'flx_{limb}']).value,
              T.cast(cc.Signal, rg[f'ext_{limb}']).value,
            ),
            dim=-1,
          ).unsqueeze(0)
        for limb in ('FL', 'FR', 'HL', 'HR')
      }

    actions = self.pattern_formation(observations, inputs)
    return actions


def modular_es(
  pattern_formation: PatternFormationNetwork,
  rhythm_generation: RhythmGenerationCircuit | None = None,
  afferent_feedback: AfferentFeedbackNetwork | None = None,
  brainstem_command: BrainstemCommandSignal | BrainstemCommandNetwork | float | list[float] |
  None = None,
  constrainer: Constrainer | bool = False,
  timestep: float = 30.,
  observation_normalizer: unflat.UnflatNormalizer | unflat.UnflatMeanStd | bool = True,
):
  if isinstance(brainstem_command, (float, int, list)):
    brainstem_command = BrainstemCommandSignal(brainstem_command)
  if observation_normalizer == True:
    observation_normalizer = unflat.UnflatMeanStd()
  if constrainer == True:
    constrainer = Constrainer()
  return unflat.UnflatActorOnly(
    actor=QuadrupedModularActor(
      pattern_formation=pattern_formation,
      rhythm_generation=rhythm_generation,
      afferent_feedback=afferent_feedback,
      brainstem_command=brainstem_command,
      constrainer=constrainer or None,
      timestep=timestep,
    ),
    observation_normalizer=observation_normalizer or None,
  )
