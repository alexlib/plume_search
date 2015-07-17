import numpy as np
from sqlalchemy import Column, ForeignKey, Sequence
from sqlalchemy import Boolean, Integer, BigInteger, Float, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

from __init__ import engine

Base = declarative_base()


class Simulation(Base):
    __tablename__ = 'simulation'

    id = Column(String(255), primary_key=True)

    n_environments = Column(Integer)
    dt = Column(Float)

    plume_structure = relationship('PlumeStructure', backref='simulation', uselist=False)
    agents = relationship('Agent', backref='simulation')
    environments = relationship('Environment', backref='simulation')


class PlumeStructure(Base):
    __tablename__ = 'plume_structure'

    id = Column(Integer, primary_key=True)

    simulation_id = Column(String(255), ForeignKey('simulation.id'))


class Agent(Base):
    __tablename__ = 'agent'

    id = Column(Integer, primary_key=True)

    simulation_id = Column(String(255), ForeignKey('simulation.id'))

    trials = relationship('Trial', backref='agent')


class Environment(Base):
    __tablename__ = 'environment'

    id = Column(Integer, primary_key=True)

    simulation_id = Column(String(255), ForeignKey('simulation.id'))

    plume_variable_sets = relationship('PlumeVariableSet', backref='environment')
    trials = relationship('Trial', backref='environment')


class PlumeVariableSet(Base):
    __tablename__ = 'plume_variable_set'

    id = Column(Integer, primary_key=True)

    src_x = Column(Float)
    src_y = Column(Float)
    src_z = Column(Float)

    environment_id = Column(Integer, ForeignKey('environment.id'))


class Trial(Base):
    __tablename__ = 'trial'

    id = Column(Integer, primary_key=True)

    plume_detected = Column(Boolean)
    time_till_detection = Column(Float)
    position_detected_x = Column(Float)
    position_detected_y = Column(Float)
    position_detected_z = Column(Float)

    agent_id = Column(Integer, ForeignKey('agent.id'))
    environment_id = Column(Integer, ForeignKey('environment.id'))


class AgentParamSetLinear(Base):
    __tablename__ = 'agent_param_set_linear'

    id = Column(Integer, primary_key=True)

    speed = Column(Float)
    theta = Column(Float)

    agent_id = Column(Integer, ForeignKey('agent.id'))
    agent = relationship('Agent', backref=backref('param_set_linear', uselist=False))


class PlumeParamSetGaussian2D(Base):
    __tablename__ = 'plume_param_set_gaussian_2d'

    id = Column(Integer, primary_key=True)

    diffusivity = Column(Float)
    wind_speed = Column(Float)
    emission_rate = Column(Float)

    plume_id = Column(Integer, ForeignKey('plume.id'))