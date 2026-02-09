"""
Main class for fitting scaffolds.
"""

import json

from cmlibs.maths.vectorops import add, mult, sub
from cmlibs.utils.zinc.field import (
    assignFieldParameters, createFieldFiniteElementClone, getGroupList, findOrCreateFieldFiniteElement,
    find_or_create_field_stored_mesh_location, getUniqueFieldName, orphanFieldByName, create_jacobian_determinant_field)
from cmlibs.utils.zinc.finiteelement import (
    evaluate_field_nodeset_range, findNodeWithName, get_scalar_field_minimum_in_mesh)
from cmlibs.utils.zinc.group import match_fitting_group_names
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.utils.zinc.mesh import element_or_ancestor_is_in_mesh
from cmlibs.utils.zinc.region import copy_fitting_data
from cmlibs.zinc.context import Context
from cmlibs.zinc.element import Elementbasis, Elementfieldtemplate
from cmlibs.zinc.field import Field, FieldFindMeshLocation, FieldGroup
from cmlibs.zinc.region import Region
from cmlibs.zinc.result import RESULT_OK, RESULT_WARNING_PART_DONE

from scaffoldfitter.fitterexceptions import FitterModelCoordinateField
from scaffoldfitter.fitterstep import FitterStep
from scaffoldfitter.fitterstepconfig import FitterStepConfig
from scaffoldfitter.fitterstepfit import FitterStepFit


def _next_available_identifier(node_set, candidate):
    node = node_set.findNodeByIdentifier(candidate)
    while node.isValid():
        candidate += 1
        node = node_set.findNodeByIdentifier(candidate)
    return candidate


class Fitter:

    def __init__(self, zincModelFileName: str=None, zincDataFileName: str=None, region: Region=None):
        """
        Create instance of Fitter either from model and data file names, or the model/fit region.
        :param zincModelFileName: Name of zinc file supplying model to fit, or None if supplying region.
        :param zincDataFileName: Name of zinc filed supplying data to fit to, or None if supplying region.
        :param region: Region in which to build model and perform fitting, or None if supplying file names.
        """
        self._zincModelFileName = zincModelFileName
        self._zincDataFileName = zincDataFileName
        if region:
            assert (zincModelFileName is None) and (zincDataFileName is None)
            self._context = region.getContext()
            self._region = region
            self._fieldmodule = region.getFieldmodule()
        else:
            assert region is None
            self._context = Context("Scaffoldfitter")
            self._region = None  # created by call to load()
            self._fieldmodule = None
        self._zincVersion = self._context.getVersion()[1]
        self._logger = self._context.getLogger()
        self._rawDataRegion = None
        self._modelCoordinatesField = None
        self._modelCoordinatesFieldName = None
        self._modelReferenceCoordinatesField = None
        self._modelFitGroup = None
        self._modelFitGroupName = None
        # fibre field is used to orient strain/curvature penalties. None=global axes
        self._fibreField = None
        self._fibreFieldName = None
        self._flattenGroup = None
        self._flattenGroupName = None
        self._dataCoordinatesField = None
        self._dataCoordinatesFieldName = None
        self._dataHostLocationField = None  # stored mesh location field in highest dimension mesh for all data, markers
        self._dataHostCoordinatesField = None  # embedded field giving host coordinates at data location
        self._dataDeltaField = None  # self._dataHostCoordinatesField - self._markerDataCoordinatesField
        self._dataErrorField = None  # magnitude of _dataDeltaField
        self._dataWeightField = None  # field storing vector of weights for each data, marker point in local directions
        self._activeDataGroupField = None
        self._activeDataNodesetGroup = None  # NodesetGroup containing all data and marker points involved in fit
        self._activeDataProjectionGroupFields = []
        self._activeDataProjectionMeshGroups = []  # [dimension - 1] line and surfaces projected onto
        self._dataProjectionGroupNames = []  # list of group names with data point projections defined
        self._dataProjectionNodeGroupFields = []  # [dimension - 1]
        self._dataProjectionNodesetGroups = []  # [dimension - 1]
        # field storing precalculated surface/line tangent and normal basis matrix
        # for transforming data delta vector to apply different sliding weights
        # Marker points use identity matrix
        self._dataProjectionOrientationField = None
        self._markerGroup = None
        self._markerGroupName = None
        self._markerNodeGroup = None
        self._markerLocationField = None
        self._markerNameField = None
        self._markerCoordinatesField = None
        self._markerDataGroup = None
        self._markerDataCoordinatesField = None
        self._markerDataNameField = None
        self._markerDataLocationGroupField = None
        self._markerDataLocationGroup = None
        # group containing union of strain, curvature active elements
        self._deformActiveGroupField = None
        self._deformActiveMeshGroup = None
        # group owning active elements with strain penalties
        self._strainActiveGroupField = None
        self._strainActiveMeshGroup = None
        # group owning active elements with curvature penalties
        self._curvatureActiveGroupField = None
        self._curvatureActiveMeshGroup = None
        self._strainPenaltyField = None  # field storing strain penalty as per-element constant
        self._curvaturePenaltyField = None  # field storing curvature penalty as per-element constant
        self._dataCentre = [0.0, 0.0, 0.0]
        self._dataScale = 1.0
        self._diagnosticLevel = 0
        self._groupProjectionData = {}  # map(group name) to (subgroup, projectionMeshGroup, findHighestDimension)
        # must always have an initial FitterStepConfig - which can never be removed
        self._fitterSteps = []
        fitterStep = FitterStepConfig()
        self.addFitterStep(fitterStep)

    def cleanup(self):
        self._fitterSteps = []
        self._clearFields()
        self._rawDataRegion = None
        self._fieldmodule = None
        self._region = None
        self._logger = None
        self._context = None

    SCAFFOLD_FITTER_SETTINGS_ID = "scaffold fitter settings"

    def decodeSettingsJSON(self, s: str, decoder):
        """
        Define Fitter from JSON serialisation output by encodeSettingsJSON.
        :param s: String of JSON encoded Fitter settings.
        :param decoder: decodeJSONFitterSteps(fitter, dct) for decodings FitterSteps.
        """
        # clear fitter steps and load from json. Later assert there is an initial config step
        oldFitterSteps = self._fitterSteps
        self._fitterSteps = []
        settings = json.loads(s, object_hook=lambda dct: decoder(self, dct))
        # self._fitterSteps will already be populated by decoder
        # ensure there is a first config step:
        if (len(self._fitterSteps) > 0) and isinstance(self._fitterSteps[0], FitterStepConfig):
            # field names are read (default to None), fields are found on load
            id = settings.get("id")
            if id is not None:
                assert id == self.SCAFFOLD_FITTER_SETTINGS_ID
                assert settings['version'] == '1.0.0'  # future: migrate if version changes
            self._modelCoordinatesFieldName = settings.get("modelCoordinatesField")
            self._modelFitGroupName = settings.get("modelFitGroup")
            self._fibreFieldName = settings.get("fibreField")
            self._flattenGroupName = settings.get("flattenGroup")
            self._dataCoordinatesFieldName = settings.get("dataCoordinatesField")
            self._markerGroupName = settings.get("markerGroup")
            self._diagnosticLevel = settings["diagnosticLevel"]
        else:
            self._fitterSteps = oldFitterSteps
            raise AssertionError("Missing initial config step")

    def encodeSettingsJSON(self) -> str:
        """
        :return: String JSON encoding of Fitter settings.
        """
        dct = {
            "id": self.SCAFFOLD_FITTER_SETTINGS_ID,
            "version": "1.0.0",
            "modelCoordinatesField": self._modelCoordinatesFieldName,
            "modelFitGroup": self._modelFitGroupName,
            "fibreField": self._fibreFieldName,
            "flattenGroup": self._flattenGroupName,
            "dataCoordinatesField": self._dataCoordinatesFieldName,
            "markerGroup": self._markerGroupName,
            "diagnosticLevel": self._diagnosticLevel,
            "fitterSteps": [fitterStep.encodeSettingsJSONDict() for fitterStep in self._fitterSteps]
        }
        return json.dumps(dct, sort_keys=False, indent=4)

    def getInitialFitterStepConfig(self):
        """
        Get first fitter step which must exist and be a FitterStepConfig.
        """
        return self._fitterSteps[0]

    def moveFitterStep(self, prevIndex, newIndex, modelFileNameStem):
        """
        Move fitter step from its previous index to a new index in the sequence, to change the order of steps.
        If a fitter step that has been run is affected by the change, the model is reloaded and no fitter steps
        after initial config are run. Can't move to/from index 0 which is initial config step.
        :param prevIndex: Previous index of step to be moved, 1 <= index < number of fitter steps.
        :param newIndex: New index for that step, 1 <= index < number of fitter steps.
        :param modelFileNameStem: File name stem for writing intermediate model files.
        :return: Boolean True if model is reloaded or False if not, index of end step.
        """
        assert 0 < prevIndex < len(self._fitterSteps)
        assert 0 < newIndex < len(self._fitterSteps)
        endStep = self._fitterSteps[0]
        for step in self._fitterSteps:
            if step.hasRun():
                endStep = step
        fitterStep = self._fitterSteps[prevIndex]
        # Switch position
        self._fitterSteps.insert(newIndex, self._fitterSteps.pop(prevIndex))
        beforeFitterStep = self._fitterSteps[newIndex - 1]
        afterFitterStep = self._fitterSteps[newIndex + 1] if newIndex < len(self._fitterSteps) - 1 else None
        isMovingStepRun = fitterStep.hasRun()
        isAfterStepRun = afterFitterStep.hasRun() if afterFitterStep else False
        isBeforeStepRun = beforeFitterStep.hasRun()
        if not isMovingStepRun and not isAfterStepRun:
            return False, self._fitterSteps.index(endStep)
        endStep = self._fitterSteps[0]
        return self.run(endStep, modelFileNameStem, True), self._fitterSteps.index(endStep)

    def getInheritFitterStep(self, refFitterStep: FitterStep):
        """
        Get last FitterStep of same type as refFitterStep or None if
        refFitterStep is the first.
        """
        refType = type(refFitterStep)
        for index in range(self._fitterSteps.index(refFitterStep) - 1, -1, -1):
            if type(self._fitterSteps[index]) is refType:
                return self._fitterSteps[index]
        return None

    def getInheritFitterStepConfig(self, refFitterStep: FitterStep):
        """
        Get last FitterStepConfig applicable to refFitterStep or None if
        refFitterStep is the first.
        """
        for index in range(self._fitterSteps.index(refFitterStep) - 1, -1, -1):
            if isinstance(self._fitterSteps[index], FitterStepConfig):
                return self._fitterSteps[index]
        return None

    def getActiveFitterStepConfig(self, refFitterStep: FitterStep):
        """
        Get latest FitterStepConfig applicable to refFitterStep.
        Can be itself.
        """
        for index in range(self._fitterSteps.index(refFitterStep), -1, -1):
            if isinstance(self._fitterSteps[index], FitterStepConfig):
                return self._fitterSteps[index]
        raise AssertionError("getActiveFitterStepConfig.  Could not find config.")

    def addFitterStep(self, fitterStep: FitterStep, refFitterStep=None):
        """
        :param fitterStep: FitterStep to add.
        :param refFitterStep: FitterStep to insert after, or None to append.
        """
        assert fitterStep.getFitter() is None
        if refFitterStep:
            self._fitterSteps.insert(self._fitterSteps.index(refFitterStep) + 1, fitterStep)
        else:
            self._fitterSteps.append(fitterStep)
        fitterStep.setFitter(self)

    def removeFitterStep(self, fitterStep: FitterStep):
        """
        Remove fitterStep from Fitter.
        :param fitterStep: FitterStep to remove. Must not be initial config.
        :return: Next FitterStep after fitterStep, or previous if None.
        """
        assert fitterStep is not self.getInitialFitterStepConfig()
        index = self._fitterSteps.index(fitterStep)
        self._fitterSteps.remove(fitterStep)
        fitterStep.setFitter(None)
        if index >= len(self._fitterSteps):
            index = -1
        return self._fitterSteps[index]

    def _clearFields(self):
        self._modelCoordinatesField = None
        self._modelReferenceCoordinatesField = None
        self._modelFitGroup = None
        self._fibreField = None
        self._flattenGroup = None
        self._dataCoordinatesField = None
        self._dataHostLocationField = None
        self._dataHostCoordinatesField = None
        self._dataDeltaField = None
        self._dataErrorField = None
        self._dataWeightField = None
        self._activeDataGroupField = None
        self._activeDataNodesetGroup = None
        self._activeDataProjectionGroupFields = []
        self._activeDataProjectionMeshGroups = []
        self._dataProjectionGroupNames = []
        self._dataProjectionNodeGroupFields = []
        self._dataProjectionNodesetGroups = []
        self._dataProjectionOrientationField = None
        self._markerGroup = None
        self._markerNodeGroup = None
        self._markerLocationField = None
        self._markerNameField = None
        self._markerCoordinatesField = None
        self._markerDataGroup = None
        self._markerDataCoordinatesField = None
        self._markerDataNameField = None
        self._markerDataLocationGroupField = None
        self._markerDataLocationGroup = None
        self._deformActiveGroupField = None
        self._deformActiveMeshGroup = None
        self._strainActiveGroupField = None
        self._strainActiveMeshGroup = None
        self._curvatureActiveGroupField = None
        self._curvatureActiveMeshGroup = None
        self._strainPenaltyField = None
        self._curvaturePenaltyField = None
        self._groupProjectionData = {}

    def load(self):
        """
        Read model and data and define fit fields and data.
        Can call again to reset fit, after parameters have changed.
        Must not call this function if model and data files not supplied to constructor!
        """
        assert self._zincModelFileName and self._zincDataFileName
        self._clearFields()
        self._region = self._context.createRegion()
        self._region.setName("model_region")
        self._fieldmodule = self._region.getFieldmodule()
        self._rawDataRegion = self._region.createChild("raw_data")
        self._loadModel()
        self._loadData()
        self.defineDataProjectionFields()
        self.initializeFit()

    def initializeFit(self):
        """
        Call after model and data are in memory, to calculate data range, mark any existing
        fitter steps as not run, and run the initial config step to calculate initial data projections.
        """
        # Get centre and scale of data coordinates to manage fitting tolerances and steps.
        datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        minimums, maximums = evaluate_field_nodeset_range(self._dataCoordinatesField, datapoints)
        self._dataCentre = mult(add(minimums, maximums), 0.5)
        self._dataScale = max(sub(maximums, minimums))
        if self._diagnosticLevel > 0:
            print("Load data: data coordinates centre ", self._dataCentre)
            print("Load data: data coordinates scale ", self._dataScale)
        for step in self._fitterSteps:
            step.setHasRun(False)
        self._fitterSteps[0].run()  # initial config step will calculate data projections

    def getDataCentre(self):
        """
        :return: Precalculated centre of data on [ x, y, z].
        """
        return self._dataCentre

    def getDataScale(self):
        """
        :return: Precalculated maximum span of data on x, y, or z.
        """
        return self._dataScale

    def defineCommonMeshFields(self):
        """
        Defines fields for storing per-element strain and curvature penalties
        plus active mesh groups for each.
        """
        mesh = self.getHighestDimensionMesh()
        meshName = mesh.getName()
        meshDimension = mesh.getDimension()
        with ChangeManager(self._fieldmodule):
            coordinatesCount = self._modelCoordinatesField.getNumberOfComponents()
            # Future issue: call this again if coordinates field changes in number of components
            self._strainPenaltyField = findOrCreateFieldFiniteElement(
                self._fieldmodule, "strain_penalty", components_count=meshDimension*meshDimension)
            self._curvaturePenaltyField = findOrCreateFieldFiniteElement(
                self._fieldmodule, "curvature_penalty", components_count=coordinatesCount*meshDimension*meshDimension)
            activeGroupFields = []
            activeMeshGroups = []
            for defname in ["deform", "strain", "curvature"]:
                activeGroupName = defname + "_active_group." + meshName
                activeGroupField = self._fieldmodule.findFieldByName(activeGroupName).castGroup()
                if not activeGroupField.isValid():
                    activeGroupField = self._fieldmodule.createFieldGroup()
                    activeGroupField.setName(activeGroupName)
                activeGroupFields.append(activeGroupField)
                activeMeshGroups.append(activeGroupField.getOrCreateMeshGroup(mesh))
            self._deformActiveGroupField, self._strainActiveGroupField, self._curvatureActiveGroupField = \
                activeGroupFields
            self._deformActiveMeshGroup, self._strainActiveMeshGroup, self._curvatureActiveMeshGroup = activeMeshGroups
            # define storage for penalty fields on all elements of mesh
            elementtemplate = mesh.createElementtemplate()
            constantBasis = self._fieldmodule.createElementbasis(meshDimension, Elementbasis.FUNCTION_TYPE_CONSTANT)
            eft = mesh.createElementfieldtemplate(constantBasis)
            eft.setParameterMappingMode(Elementfieldtemplate.PARAMETER_MAPPING_MODE_ELEMENT)
            elementtemplate.defineField(self._strainPenaltyField, -1, eft)
            elementtemplate.defineField(self._curvaturePenaltyField, -1, eft)
            elemIter = mesh.createElementiterator()
            fieldcache = self._fieldmodule.createFieldcache()
            element = elemIter.next()
            zeroValues = [0.0] * 27
            while element.isValid():
                element.merge(elementtemplate)
                fieldcache.setElement(element)
                self._strainPenaltyField.assignReal(fieldcache, zeroValues)
                self._curvaturePenaltyField.assignReal(fieldcache, zeroValues)
                element = elemIter.next()

    def getStrainPenaltyField(self):
        return self._strainPenaltyField

    def getCurvaturePenaltyField(self):
        return self._curvaturePenaltyField

    def _loadModel(self):
        result = self._region.readFile(self._zincModelFileName)
        assert result == RESULT_OK, "Failed to load model file" + str(self._zincModelFileName)
        self._discoverModelCoordinatesField()
        self._discoverModelFitGroup()
        self._discoverFibreField()
        self._discoverFlattenGroup()
        self.defineCommonMeshFields()

    def _defineCommonDataFields(self):
        """
        Defines self._dataHostCoordinatesField to gives the value of self._modelCoordinatesField at
        embedded location self._dataHostLocationField.
        Need to call again if self._modelCoordinatesField is changed.
        """
        # need to store all data + marker locations in top-level elements for NEWTON objective
        # in future may want to support mixed dimension top-level elements
        if not (self._modelCoordinatesField and self._dataCoordinatesField):
            return  # on first load, can't call until setModelCoordinatesField and setDataCoordinatesField
        with ChangeManager(self._fieldmodule):
            mesh = self.getHighestDimensionMesh()
            datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
            self._dataHostLocationField = find_or_create_field_stored_mesh_location(
                self._fieldmodule, mesh, "data_location_" + mesh.getName(), managed=False)
            self._dataHostCoordinatesField = self._fieldmodule.createFieldEmbedded(
                self._modelCoordinatesField, self._dataHostLocationField)
            self._dataHostCoordinatesField.setName(getUniqueFieldName(self._fieldmodule, "data_host_coordinates"))
            self._dataDeltaField = self._dataHostCoordinatesField - self._dataCoordinatesField
            self._dataDeltaField.setName(getUniqueFieldName(self._fieldmodule, "data_delta"))
            self._dataErrorField = self._fieldmodule.createFieldMagnitude(self._dataDeltaField)
            self._dataErrorField.setName(getUniqueFieldName(self._fieldmodule, "data_error"))
            # store weights per-point so can maintain variable weights for marker and data by group
            coordinatesCount = self._modelCoordinatesField.getNumberOfComponents()
            self._dataWeightField = \
                findOrCreateFieldFiniteElement(self._fieldmodule, "data_weight", components_count=coordinatesCount)
            activeDataName = "active_data.datapoints"
            self._activeDataGroupField = self._fieldmodule.findFieldByName(activeDataName).castGroup()
            if not self._activeDataGroupField.isValid():
                self._activeDataGroupField = self._fieldmodule.createFieldGroup()
                self._activeDataGroupField.setName(activeDataName)
            self._activeDataNodesetGroup = self._activeDataGroupField.getOrCreateNodesetGroup(datapoints)
            self._activeDataProjectionGroupFields = []
            self._activeDataProjectionMeshGroups = []
            for d in range(2):
                mesh = self.getMesh(d + 1)
                activeDataProjectionGroupName = "active_data." + mesh.getName()
                activeDataProjectionGroupField = \
                    self._fieldmodule.findFieldByName(activeDataProjectionGroupName).castGroup()
                if not activeDataProjectionGroupField.isValid():
                    activeDataProjectionGroupField = self._fieldmodule.createFieldGroup()
                    activeDataProjectionGroupField.setName(activeDataProjectionGroupName)
                self._activeDataProjectionGroupFields.append(activeDataProjectionGroupField)
                self._activeDataProjectionMeshGroups.append(activeDataProjectionGroupField.getOrCreateMeshGroup(mesh))

    def _loadData(self):
        """
        Load zinc data file into self._rawDataRegion & transfer as data points to fit region.
        Rename data groups to exactly match model groups where they differ by case and whitespace only.
        """
        result = self._rawDataRegion.readFile(self._zincDataFileName)
        assert result == RESULT_OK, "Failed to load data file " + str(self._zincDataFileName)
        data_fieldmodule = self._rawDataRegion.getFieldmodule()
        with ChangeManager(data_fieldmodule):
            match_fitting_group_names(data_fieldmodule, self._fieldmodule,
                                      log_diagnostics=self.getDiagnosticLevel() > 0)
            try:
                copy_fitting_data(self._region, self._rawDataRegion)
            except:
                self.print_log()
        self._discoverDataCoordinatesField()
        self._discoverMarkerGroup()

    def run(self, endStep=None, modelFileNameStem=None, reorder=False):
        """
        Run either all remaining fitter steps or up to specified end step.
        Only call this to run to a previous step if Fitter is working with model and data files;
        see __init__().
        :param endStep: Last fitter step to run, or None to run all.
        :param modelFileNameStem: File name stem for writing intermediate model files.
        :param reorder: Reload if reordering.
        :return: True if reloaded (so scene changed), False if not.
        """
        if not endStep:
            endStep = self._fitterSteps[-1]
        endIndex = self._fitterSteps.index(endStep)
        # reload only if necessary
        if (endStep.hasRun() and (endIndex < (len(self._fitterSteps) - 1)) and self._fitterSteps[endIndex + 1].hasRun()
                or reorder):
            # re-load to get back to current state
            self.load()
            for index in range(1, endIndex + 1):
                self._fitterSteps[index].run(modelFileNameStem + str(index) if modelFileNameStem else None)
            return True
        if endIndex == 0:
            endStep.run()  # force re-run initial config
        else:
            # run from current point up to step
            for index in range(1, endIndex + 1):
                if not self._fitterSteps[index].hasRun():
                    self._fitterSteps[index].run(modelFileNameStem + str(index) if modelFileNameStem else None)
        return False

    def getDataCoordinatesField(self):
        return self._dataCoordinatesField

    def setDataCoordinatesField(self, dataCoordinatesField: Field):
        if (self._dataCoordinatesField is not None) and (dataCoordinatesField == self._dataCoordinatesField):
            return
        finiteElementField = dataCoordinatesField.castFiniteElement()
        assert finiteElementField.isValid() and (finiteElementField.getNumberOfComponents() == 3)
        self._dataCoordinatesFieldName = dataCoordinatesField.getName()
        self._dataCoordinatesField = finiteElementField
        self._defineCommonDataFields()
        self._calculateMarkerDataLocations()  # needed to assign to self._dataCoordinatesField

    def setDataCoordinatesFieldByName(self, dataCoordinatesFieldName):
        self.setDataCoordinatesField(self._fieldmodule.findFieldByName(dataCoordinatesFieldName))

    def _discoverDataCoordinatesField(self):
        """
        Choose default dataCoordinates field.
        """
        self._dataCoordinatesField = None
        field = None

        if self._dataCoordinatesFieldName:
            field = self._fieldmodule.findFieldByName(self._dataCoordinatesFieldName)
        if not (field and field.isValid()):
            datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
            datapoint = datapoints.createNodeiterator().next()
            if datapoint.isValid():
                fieldcache = self._fieldmodule.createFieldcache()
                fieldcache.setNode(datapoint)
                fielditer = self._fieldmodule.createFielditerator()
                field = fielditer.next()
                while field.isValid():
                    if field.isTypeCoordinate() and (field.getNumberOfComponents() == 3) and \
                            (field.castFiniteElement().isValid()):
                        if field.isDefinedAtLocation(fieldcache):
                            break
                    field = fielditer.next()
                else:
                    field = None
        self.setDataCoordinatesField(field)

    def getMarkerGroup(self):
        return self._markerGroup

    def setMarkerGroup(self, markerGroup: Field):
        self._markerGroup = None
        self._markerGroupName = None
        self._markerNodeGroup = None
        self._markerLocationField = None
        self._markerCoordinatesField = None
        self._markerNameField = None
        self._markerDataGroup = None
        self._markerDataCoordinatesField = None
        self._markerDataNameField = None
        self._markerDataLocationGroupField = None
        self._markerDataLocationGroup = None
        if not (markerGroup and markerGroup.isValid()):
            return
        fieldGroup = markerGroup.castGroup()
        assert fieldGroup.isValid()
        self._markerGroup = fieldGroup
        self._markerGroupName = markerGroup.getName()
        nodes = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self._markerNodeGroup = self._markerGroup.getNodesetGroup(nodes)
        if self._markerNodeGroup.isValid():
            node = self._markerNodeGroup.createNodeiterator().next()
            if node.isValid():
                fieldcache = self._fieldmodule.createFieldcache()
                fieldcache.setNode(node)
                fielditer = self._fieldmodule.createFielditerator()
                field = fielditer.next()
                while field.isValid():
                    if field.isDefinedAtLocation(fieldcache):
                        if (not self._markerLocationField) and field.castStoredMeshLocation().isValid():
                            self._markerLocationField = field
                        elif (not self._markerNameField) and (field.getValueType() == Field.VALUE_TYPE_STRING):
                            self._markerNameField = field
                    field = fielditer.next()
                self._updateMarkerCoordinatesField()
        else:
            self._markerNodeGroup = None
        datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self._markerDataGroup = self._markerGroup.getNodesetGroup(datapoints)
        if self._markerDataGroup.isValid():
            datapoint = self._markerDataGroup.createNodeiterator().next()
            if datapoint.isValid():
                fieldcache = self._fieldmodule.createFieldcache()
                fieldcache.setNode(datapoint)
                fielditer = self._fieldmodule.createFielditerator()
                field = fielditer.next()
                while field.isValid():
                    if field.isDefinedAtLocation(fieldcache):
                        if (not self._markerDataCoordinatesField) and field.isTypeCoordinate() and \
                                (field.getNumberOfComponents() == 3) and (field.castFiniteElement().isValid()):
                            self._markerDataCoordinatesField = field
                        elif (not self._markerDataNameField) and (field.getValueType() == Field.VALUE_TYPE_STRING):
                            self._markerDataNameField = field
                    field = fielditer.next()
        else:
            self._markerDataGroup = None
        self._calculateMarkerDataLocations()

    def assignDataWeights(self, fitterStepFit: FitterStepFit):
        """
        Assign values of the weight field for all data and marker points.
        """
        # Future: divide by linear data scale?
        # Future: divide by number of data points?
        coordinatesCount = self._modelCoordinatesField.getNumberOfComponents()
        with ChangeManager(self._fieldmodule):
            for groupName in self._dataProjectionGroupNames:
                group = self._fieldmodule.findFieldByName(groupName).castGroup()
                if not group.isValid():
                    continue
                dataGroup = self.getGroupDataProjectionNodesetGroup(group)
                if not dataGroup:
                    continue
                meshGroup = self.getGroupDataProjectionMeshGroup(group, fitterStepFit)[0]
                meshDimension = meshGroup.getDimension() if meshGroup else 0
                if (meshDimension < 1) or (meshDimension > 2):
                    continue
                dataWeight = fitterStepFit.getGroupDataWeight(groupName)[0]
                dataSlidingFactor = fitterStepFit.getGroupDataSlidingFactor(groupName)[0]
                slidingWeight = dataWeight * dataSlidingFactor
                if meshDimension == 1:
                    if coordinatesCount == 3:
                        orientedDataWeight = [slidingWeight, dataWeight, dataWeight]
                    elif coordinatesCount == 2:
                        orientedDataWeight = [slidingWeight, dataWeight]
                    else:  # coordinatesCount == 1:
                        orientedDataWeight = [slidingWeight]  # not expected
                else:  # meshDimension == 2:
                    if coordinatesCount == 3:
                        orientedDataWeight = [slidingWeight, slidingWeight, dataWeight]
                    else:  # coordinatesCount == 2:
                        orientedDataWeight = [slidingWeight, slidingWeight]  # not expected
                stretchOrientedDataWeight = [dataWeight] + orientedDataWeight[1:]
                weightField = self._fieldmodule.createFieldConstant(orientedDataWeight)
                dataStretch = fitterStepFit.getGroupDataStretch(groupName)[0]
                if dataStretch:
                    tangent1 = self._fieldmodule.createFieldComponent(
                        self._dataProjectionOrientationField,
                        [1, 2, 3] if (coordinatesCount == 3) else [1, 2] if (coordinatesCount == 2) else [1])
                    weightField = self._fieldmodule.createFieldIf(
                        self._fieldmodule.createFieldGreaterThan(
                            self._fieldmodule.createFieldDotProduct(self._dataDeltaField, tangent1),
                            self._fieldmodule.createFieldConstant([0.01]) *
                            self._fieldmodule.createFieldMagnitude(self._dataDeltaField)),
                        self._fieldmodule.createFieldConstant(stretchOrientedDataWeight),
                        weightField)
                    del tangent1
                if self._diagnosticLevel > 0:
                    print("group", groupName, "mesh dimension", meshDimension, "data weight", dataWeight,
                          "sliding factor", dataSlidingFactor, "stretch", dataStretch)
                fieldassignment = self._dataWeightField.createFieldassignment(weightField)
                fieldassignment.setNodeset(dataGroup)
                result = fieldassignment.assign()
                if result != RESULT_OK:
                    print("Incomplete assignment of data weight for group", groupName, "Result", result)
                del weightField
            if self._markerDataLocationGroup:
                markerWeight = fitterStepFit.getGroupDataWeight(self._markerGroupName)[0]
                orientedDataWeight = [markerWeight] * coordinatesCount
                # print("marker weight", markerWeight)
                fieldassignment = self._dataWeightField.createFieldassignment(
                    self._fieldmodule.createFieldConstant(orientedDataWeight))
                fieldassignment.setNodeset(self._markerDataLocationGroup)
                result = fieldassignment.assign()
                if result != RESULT_OK:
                    print('Incomplete assignment of marker data weight', result)
            del fieldassignment

    def assignDeformationPenalties(self, fitterStepFit: FitterStepFit):
        """
        Assign per-element strain and curvature penalty values and build
        groups of elements for which they are non-zero.
        If element is in multiple groups with values set, value for first group found is used.
        Currently applied only to elements of highest dimension.
        :return: deformActiveMeshGroup, strainActiveMeshGroup, curvatureActiveMeshGroup
        Zinc MeshGroups over which to apply penalties: combined, strain and curvature.
        """
        # Future: divide by linear data scale?
        # Future: divide by number of data points?
        # Get list of mesh groups of highest dimension with strain, curvature penalties
        mesh = self.getHighestDimensionMesh()
        meshDimension = mesh.getDimension()
        coordinatesCount = self._modelCoordinatesField.getNumberOfComponents()
        strainComponents = meshDimension * meshDimension
        curvatureComponents = coordinatesCount * meshDimension * meshDimension
        groupInfoList = []
        # add None for default group
        for group in (getGroupList(self._fieldmodule) + [None]):
            if group:
                if not group.isManaged():
                    continue  # unmanaged = internal, ignore
                meshGroup = group.getMeshGroup(mesh)
                meshGroupSize = meshGroup.getSize()
                if (not meshGroup.isValid()) or (meshGroupSize == 0):
                    continue
                groupName = group.getName()
            else:
                meshGroup = None
                groupName = None
                meshGroupSize = mesh.getSize()
            groupStrainPenalty, setLocally, inheritable = \
                fitterStepFit.getGroupStrainPenalty(groupName, strainComponents)
            groupStrainPenaltyNonZero = any((s > 0.0) for s in groupStrainPenalty)
            groupStrainSet = setLocally or ((setLocally is False) and inheritable)
            groupCurvaturePenalty, setLocally, inheritable = \
                fitterStepFit.getGroupCurvaturePenalty(groupName, curvatureComponents)
            groupCurvaturePenaltyNonZero = any((s > 0.0) for s in groupCurvaturePenalty)
            groupCurvatureSet = setLocally or ((setLocally is False) and inheritable)
            groupInfo = (group, groupName, meshGroup, meshGroupSize, groupStrainPenalty, groupStrainPenaltyNonZero,
                         groupStrainSet, groupCurvaturePenalty, groupCurvaturePenaltyNonZero, groupCurvatureSet)
            for index, tmpGroupInfo in enumerate(groupInfoList):
                if meshGroupSize < tmpGroupInfo[3]:
                    groupInfoList.insert(index, groupInfo)
                    break
            else:
                groupInfoList.append(groupInfo)
        with ChangeManager(self._fieldmodule):
            self._deformActiveMeshGroup.removeAllElements()
            self._strainActiveMeshGroup.removeAllElements()
            self._curvatureActiveMeshGroup.removeAllElements()
            elementIter = mesh.createElementiterator()
            element = elementIter.next()
            fieldcache = self._fieldmodule.createFieldcache()
            while element.isValid():
                fieldcache.setElement(element)
                strainPenalty = None
                strainPenaltyNonZero = False
                curvaturePenalty = None
                curvaturePenaltyNonZero = False
                for (group, groupName, meshGroup, meshGroupSize, groupStrainPenalty, groupStrainPenaltyNonZero,
                     groupStrainSet, groupCurvaturePenalty, groupCurvaturePenaltyNonZero, groupCurvatureSet) \
                        in groupInfoList:
                    if (not group) or meshGroup.containsElement(element):
                        if (not strainPenalty) and (groupStrainSet or (not group)):
                            strainPenalty = groupStrainPenalty
                            strainPenaltyNonZero = groupStrainPenaltyNonZero
                        if (not curvaturePenalty) and (groupCurvatureSet or (not group)):
                            curvaturePenalty = groupCurvaturePenalty
                            curvaturePenaltyNonZero = groupCurvaturePenaltyNonZero
                # always assign strain, curvature penalties to clear to zero where not used
                self._strainPenaltyField.assignReal(fieldcache, strainPenalty)
                self._curvaturePenaltyField.assignReal(fieldcache, curvaturePenalty)
                if strainPenaltyNonZero:
                    self._strainActiveMeshGroup.addElement(element)
                    if self._diagnosticLevel > 1:
                        print("Element", element.getIdentifier(), "apply strain penalty", strainPenalty)
                if curvaturePenaltyNonZero:
                    self._curvatureActiveMeshGroup.addElement(element)
                    if self._diagnosticLevel > 1:
                        print("Element", element.getIdentifier(), "apply curvature penalty", curvaturePenalty)
                if strainPenaltyNonZero or curvaturePenaltyNonZero:
                    self._deformActiveMeshGroup.addElement(element)
                element = elementIter.next()
        return self._deformActiveMeshGroup, self._strainActiveMeshGroup, self._curvatureActiveMeshGroup

    def setMarkerGroupByName(self, markerGroupName):
        self.setMarkerGroup(self._fieldmodule.findFieldByName(markerGroupName))

    def getDataHostLocationField(self):
        return self._dataHostLocationField

    def getDataHostCoordinatesField(self):
        """
        :return: Field giving coordinates of projections of data points on mesh.
        """
        return self._dataHostCoordinatesField

    def getDataDeltaField(self):
        """
        :return: Field giving delta coordinates (projection coordinates - data coordinates)
        for all data points & data marker points.
        """
        return self._dataDeltaField

    def getDataErrorField(self):
        """
        :return: Field giving magnitude of data point delta field.
        """
        return self._dataErrorField

    def getDataWeightField(self):
        return self._dataWeightField

    def getActiveDataNodesetGroup(self):
        return self._activeDataNodesetGroup

    def getDataRMSAndMaximumProjectionError(self, nodesetGroup=None):
        """
        Get RMS and maximum error for all active data and marker point projections.
        No group weights are applied.
        Optional nodeset group parameter allows the user to make the calculation over a different group
        from the active group.
        :param nodesetGroup: Optional parameter to specify a particular nodeset group to make the calculation over.
        :return: RMS error value, maximum error value. Values are None if there is no data or bad fields.
        """
        calculatedNodesetGroup = nodesetGroup if nodesetGroup else self._activeDataNodesetGroup
        with ChangeManager(self._fieldmodule):
            error = self._fieldmodule.createFieldMagnitude(self._dataDeltaField)
            msError = self._fieldmodule.createFieldNodesetMeanSquares(error, calculatedNodesetGroup)
            rmsError = self._fieldmodule.createFieldSqrt(msError)
            maxError = self._fieldmodule.createFieldNodesetMaximum(error, calculatedNodesetGroup)
            fieldcache = self._fieldmodule.createFieldcache()
            rmsResult, rmsErrorValue = rmsError.evaluateReal(fieldcache, 1)
            maxResult, maxErrorValue = maxError.evaluateReal(fieldcache, 1)
            del fieldcache
            del maxError
            del rmsError
            del msError
            del error
        return rmsErrorValue if (rmsResult == RESULT_OK) else None, maxErrorValue if (maxResult == RESULT_OK) else None

    def getDataRMSAndMaximumProjectionErrorForGroup(self, groupName):
        """
        Get RMS and maximum error for the intersection of the nodeset group with 
        the given name and the active data group projections.
        If the groupName is not a valid nodeset group name or the calculation fails
        then None, None is returned.

        :param groupName: Name of group to make calculation over.
        :return: RMS error value, maximum error value.
        """
        with ChangeManager(self._fieldmodule):
            group = self._fieldmodule.findFieldByName(groupName).castGroup()
            if group.isValid():
                calculation_field = self._fieldmodule.createFieldAnd(group, self._activeDataGroupField)
                nodeset = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
                temp_dataset_group = self._fieldmodule.createFieldGroup().createNodesetGroup(nodeset)
                temp_dataset_group.addNodesConditional(calculation_field)
                result = self.getDataRMSAndMaximumProjectionError(temp_dataset_group)
                del temp_dataset_group
                del calculation_field
                return result

        return None, None

    def getLowestElementJacobian(self, mesh_group=None):
        """
        Get the information on the 3D element with the worst jacobian value (most negative).
        For a right-handed element values <= 0.0 are bad, the opposite holds for left-handed
        elements.
        Optional mesh group parameter allows the user to make the calculation over a different group
        from the whole mesh.
        :param mesh_group: Optional parameter to specify a particular mesh group to make the calculation over.
        :return: Element identifier, minimum jacobian value. Values are -1, inf if there is no data or bad fields.
        """
        with ChangeManager(self._fieldmodule):
            jacobian = create_jacobian_determinant_field(self._modelCoordinatesField, self._modelReferenceCoordinatesField)
            result = get_scalar_field_minimum_in_mesh(jacobian, mesh_group)
            del jacobian

        return result

    def getLowestElementJacobianForGroup(self, group_name):
        """
        Get the information on the 3D element with the worst Jacobian
        value (most negative) of the 3D mesh group with the given name.
        If the group_name is not a valid group name then None, None is returned.
        If the fields for the calculation of the Jacobian are invalid then
        -1, infinity is returned.

        :param group_name: Name of group to make calculation over.
        :return: Element identifier, minimum Jacobian value.
        """
        group = self._fieldmodule.findFieldByName(group_name).castGroup()
        if group.isValid():
            mesh_group = group.getMeshGroup(self.getMesh(3))
            return self.getLowestElementJacobian(mesh_group)

        return None, None

    def getMarkerDataFields(self):
        """
        Only call if markerGroup exists.
        :return: markerDataGroup, markerDataCoordinates, markerDataName
        """
        return self._markerDataGroup, self._markerDataCoordinatesField, self._markerDataNameField

    def getMarkerDataLocationFields(self):
        """
        Get fields giving marker location coordinates and delta on the data points (copied from nodes).
        Only call if markerGroup exists.
        :return: markerDataLocation, markerDataLocationCoordinates, markerDataDelta
        """
        # these are now common:
        return self._dataHostLocationField, self._dataHostCoordinatesField, self._dataDeltaField

    def getMarkerModelFields(self):
        """
        Only call if markerGroup exists.
        :return: markerNodeGroup, markerLocation, markerCoordinates, markerName
        """
        return self._markerNodeGroup, self._markerLocationField, self._markerCoordinatesField, self._markerNameField

    def _calculateMarkerDataLocations(self):
        """
        Called when markerGroup exists.
        Find matching marker mesh locations for marker data points.
        Only finds matching location where there is one datapoint and one node
        for each name in marker group.
        Defines datapoint group self._markerDataLocationGroup to contain those with locations.
        """
        self._markerDataLocationGroupField = None
        self._markerDataLocationGroup = None
        if not (self._markerDataGroup and self._markerDataNameField and self._markerNodeGroup and
                self._markerLocationField and self._markerNameField):
            return

        markerPrefix = self._markerGroupName
        # assume marker locations are in highest dimension mesh
        mesh = self.getHighestDimensionMesh()
        datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        meshDimension = mesh.getDimension()
        with ChangeManager(self._fieldmodule):
            self._markerDataLocationGroupField = self._fieldmodule.createFieldGroup()
            self._markerDataLocationGroupField.setName(
                getUniqueFieldName(self._fieldmodule, markerPrefix + "_data_location_group"))
            self._markerDataLocationGroup = self._markerDataLocationGroupField.createNodesetGroup(datapoints)
            nodetemplate = self._markerDataGroup.createNodetemplate()
            nodetemplate.defineField(self._dataHostLocationField)
            coordinatesCount = self._markerDataCoordinatesField.getNumberOfComponents()
            defineDataCoordinates = self._markerDataCoordinatesField != self._dataCoordinatesField
            if defineDataCoordinates:
                # define dataCoordinates on marker points for combined objective, and assign below
                assert self._dataCoordinatesField.isValid()
                nodetemplate.defineField(self._dataCoordinatesField)
            # need to define storage for marker data weight, but don't assign here
            nodetemplate.defineField(self._dataWeightField)
            self._defineDataProjectionOrientationField()
            nodetemplate.defineField(self._dataProjectionOrientationField)
            findMarkerLocation = None
            modelFitMeshGroup = None
            undefineNodetemplate = None
            if self._modelFitGroup:
                # following needed for re-finding marker point locations on boundary of model fit group
                modelFitMeshGroup = self._modelFitGroup.getMeshGroup(mesh)
                findMarkerLocation = self._fieldmodule.createFieldFindMeshLocation(
                    self._fieldmodule.createFieldEmbedded(
                        self._modelReferenceCoordinatesField, self._markerLocationField),
                    self._modelReferenceCoordinatesField, modelFitMeshGroup)
                undefineNodetemplate = self._markerDataGroup.createNodetemplate()
                undefineNodetemplate.undefineField(self._dataHostLocationField)
                if defineDataCoordinates:
                    undefineNodetemplate.undefineField(self._dataCoordinatesField)
            fieldcache = self._fieldmodule.createFieldcache()
            datapointIter = self._markerDataGroup.createNodeiterator()
            datapoint = datapointIter.next()

            # data projection orientation field is just the identity matrix for marker points
            dataProjectionOrientation = \
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] if coordinatesCount == 3 else \
                [1.0, 0.0, 0.0, 1.0] if coordinatesCount == 2 else \
                [1.0]
            while datapoint.isValid():
                fieldcache.setNode(datapoint)
                name = self._markerDataNameField.evaluateString(fieldcache)
                # if this is the only datapoint with name:
                if name and findNodeWithName(self._markerDataGroup, self._markerDataNameField, name, ignore_case=True,
                                             strip_whitespace=True):
                    result, dataCoordinates = \
                        self._markerDataCoordinatesField.evaluateReal(fieldcache, coordinatesCount)
                    node = findNodeWithName(self._markerNodeGroup, self._markerNameField, name, ignore_case=True,
                                            strip_whitespace=True)
                    if (result == RESULT_OK) and node:
                        fieldcache.setNode(node)
                        element, xi = self._markerLocationField.evaluateMeshLocation(fieldcache, meshDimension)
                        if element.isValid():
                            if self._modelFitGroup and not modelFitMeshGroup.containsElement(element):
                                # ensure marker points on boundary of model fit group are moved to elements in it
                                element, xi = findMarkerLocation.evaluateMeshLocation(fieldcache, meshDimension)
                                if not element.isValid():
                                    datapoint.merge(undefineNodetemplate)
                        if element.isValid():  # ignore markers not in model fit group
                            datapoint.merge(nodetemplate)
                            fieldcache.setNode(datapoint)
                            self._dataHostLocationField.assignMeshLocation(fieldcache, element, xi)
                            if defineDataCoordinates:
                                self._dataCoordinatesField.assignReal(fieldcache, dataCoordinates)
                            self._dataProjectionOrientationField.assignReal(fieldcache, dataProjectionOrientation)
                            self._markerDataLocationGroup.addNode(datapoint)
                datapoint = datapointIter.next()
            del fieldcache
            del findMarkerLocation
            # ensure activeDataNodeset only contains active marker points
            self._activeDataNodesetGroup.removeNodesConditional(self._markerGroup)
            self._activeDataNodesetGroup.addNodesConditional(self._markerDataLocationGroupField)

        # Warn about marker points without a location in model
        markerDataGroupSize = self._markerDataGroup.getSize()
        markerDataLocationGroupSize = self._markerDataLocationGroup.getSize()
        markerNodeGroupSize = self._markerNodeGroup.getSize()
        if self.getDiagnosticLevel() > 0:
            print(str(markerDataLocationGroupSize) + " of " + str(markerDataGroupSize) +
                  " marker data points have model locations")
            if markerDataLocationGroupSize < markerDataGroupSize:
                print("Warning: Only " + str(markerDataLocationGroupSize) +
                      " of " + str(markerDataGroupSize) + " marker data points have model locations")
            if markerDataLocationGroupSize < markerNodeGroupSize:
                print("Warning: Only " + str(markerDataLocationGroupSize) +
                      " of " + str(markerNodeGroupSize) + " marker model locations used")

    def _discoverMarkerGroup(self):
        self._markerGroup = None
        self._markerNodeGroup = None
        self._markerLocationField = None
        self._markerNameField = None
        self._markerCoordinatesField = None
        markerGroupName = self._markerGroupName if self._markerGroupName else "marker"
        markerGroup = self._fieldmodule.findFieldByName(markerGroupName).castGroup()
        if not markerGroup.isValid():
            markerGroup = None
        self.setMarkerGroup(markerGroup)

    def _updateMarkerCoordinatesField(self):
        if self._modelCoordinatesField and self._markerLocationField:
            with ChangeManager(self._fieldmodule):
                markerPrefix = self._markerGroup.getName()
                self._markerCoordinatesField = \
                    self._fieldmodule.createFieldEmbedded(self._modelCoordinatesField, self._markerLocationField)
                self._markerCoordinatesField.setName(
                    getUniqueFieldName(self._fieldmodule, markerPrefix + "_coordinates"))
        else:
            self._markerCoordinatesField = None

    def getModelCoordinatesField(self):
        return self._modelCoordinatesField

    def getModelReferenceCoordinatesField(self):
        return self._modelReferenceCoordinatesField

    def setModelCoordinatesField(self, modelCoordinatesField: Field):
        if (self._modelCoordinatesField is not None) and (modelCoordinatesField == self._modelCoordinatesField):
            return
        finiteElementField = modelCoordinatesField.castFiniteElement()
        mesh = self.getHighestDimensionMesh()
        assert finiteElementField.isValid() and (mesh.getDimension() <= finiteElementField.getNumberOfComponents() <= 3)
        self._modelCoordinatesField = finiteElementField
        self._modelCoordinatesFieldName = modelCoordinatesField.getName()
        modelReferenceCoordinatesFieldName = "reference_" + self._modelCoordinatesField.getName()
        orphanFieldByName(self._fieldmodule, modelReferenceCoordinatesFieldName)
        self._modelReferenceCoordinatesField = \
            createFieldFiniteElementClone(self._modelCoordinatesField, modelReferenceCoordinatesFieldName)
        self._defineCommonDataFields()
        self._updateMarkerCoordinatesField()

    def setModelCoordinatesFieldByName(self, modelCoordinatesFieldName):
        self.setModelCoordinatesField(self._fieldmodule.findFieldByName(modelCoordinatesFieldName))

    def _find_first_coordinate_type_field(self):
        field = None

        mesh = self.getHighestDimensionMesh()
        element = mesh.createElementiterator().next()
        if element.isValid():
            fieldcache = self._fieldmodule.createFieldcache()
            fieldcache.setElement(element)
            fielditer = self._fieldmodule.createFielditerator()
            field = fielditer.next()
            while field.isValid():
                if field.isTypeCoordinate() and (field.getNumberOfComponents() == 3) and \
                        (field.castFiniteElement().isValid()):
                    if field.isDefinedAtLocation(fieldcache):
                        break
                field = fielditer.next()
            else:
                field = None

        return field

    def _discoverModelCoordinatesField(self):
        """
        Choose default modelCoordinates field.
        """
        self._modelCoordinatesField = None
        self._modelReferenceCoordinatesField = None
        field = None
        if self._modelCoordinatesFieldName:
            field = self._fieldmodule.findFieldByName(self._modelCoordinatesFieldName)

        if field is None or not field.isValid():
            field = self._find_first_coordinate_type_field()

        if field and field.isValid():
            self.setModelCoordinatesField(field)
        else:
            raise FitterModelCoordinateField("No coordinate field found for model.")

    def getModelFitGroup(self):
        return self._modelFitGroup

    def setModelFitGroup(self, modelFitGroup: Field):
        """
        Set subset of model to fit over. Must be a group field with a mesh group for
        mesh of highest dimension in model
        :param modelFitGroup: Zinc group or None for whole mesh.
        """
        if (self._modelFitGroup is not None) and (modelFitGroup == self._modelFitGroup):
            return
        self._groupProjectionData = {}  # as this affects intersection groups
        fieldGroup = modelFitGroup.castGroup() if modelFitGroup else None
        assert (fieldGroup is None) or fieldGroup.isValid()
        if fieldGroup:
            meshGroup = fieldGroup.getMeshGroup(self.getHighestDimensionMesh())
            if (not meshGroup.isValid()) or (meshGroup.getSize() == 0):
                print("Cannot set model fit group", modelFitGroup.getName(), "as empty mesh group at highest dimension")
                return
        self._modelFitGroup = fieldGroup
        self._modelFitGroupName = modelFitGroup.getName() if modelFitGroup else None
        self._calculateMarkerDataLocations()  # needed to move or ignore markers not in model fit group

    def setModelFitGroupByName(self, modelFitGroupName):
        self.setModelFitGroup(self._fieldmodule.findFieldByName(modelFitGroupName))

    def _discoverModelFitGroup(self):
        """
        Discover modelFitGroup from set name.
        """
        self._modelFitGroup = None
        if self._modelFitGroupName:
            field = self._fieldmodule.findFieldByName(self._modelFitGroupName)
            if field.isValid():
                self.setModelFitGroup(field)

    def getFibreField(self):
        return self._fibreField

    def setFibreField(self, fibreField: Field):
        """
        Set field used to orient strain and curvature penalties relative to element.
        :param fibreField: Fibre angles field available on elements, or None to use
        global x, y, z axes.
        """
        assert (fibreField is None) or \
            ((fibreField.getValueType() == Field.VALUE_TYPE_REAL) and (fibreField.getNumberOfComponents() <= 3)), \
            "Scaffoldfitter: Invalid fibre field"
        self._fibreField = fibreField
        self._fibreFieldName = fibreField.getName() if fibreField else None

    def _discoverFibreField(self):
        """
        Find field used to orient strain and curvature penalties, if any.
        """
        self._fibreField = None
        fibreField = None
        # guarantee a zero fibres field exists
        zeroFibreFieldName = "zero fibres"
        zeroFibreField = self._fieldmodule.findFieldByName(zeroFibreFieldName)
        if not zeroFibreField.isValid():
            with ChangeManager(self._fieldmodule):
                zeroFibreField = self._fieldmodule.createFieldConstant([0.0, 0.0, 0.0])
                zeroFibreField.setName(zeroFibreFieldName)
                zeroFibreField.setManaged(True)
        if self._fibreFieldName:
            fibreField = self._fieldmodule.findFieldByName(self._fibreFieldName)
        if not (fibreField and fibreField.isValid()):
            fibreField = None  # in future, could be zeroFibreField?
        self.setFibreField(fibreField)

    def getFlattenGroup(self):
        return self._flattenGroup

    def setFlattenGroup(self, flattenGroup: Field):
        """
        Set group to constrain to z = 0 (or y = 0 for 2-D coordinates).
        Data weight for that group is used as weighting on integral.
        :param flattenGroup: Zinc group or None for whole mesh.
        """
        assert (flattenGroup is None) or flattenGroup.castGroup().isValid(), \
            "Scaffoldfitter: Invalid flatten group"
        fieldGroup = flattenGroup.castGroup() if flattenGroup else None
        assert (fieldGroup is None) or fieldGroup.isValid()
        self._flattenGroup = fieldGroup
        self._flattenGroupName = flattenGroup.getName() if flattenGroup else None

    def setFlattenGroupByName(self, flattenGroupName):
        self.setFlattenGroup(self._fieldmodule.findFieldByName(flattenGroupName))

    def _discoverFlattenGroup(self):
        """
        Discover flattenGroup from set name.
        """
        self._flattenGroup = None
        if self._flattenGroupName:
            field = self._fieldmodule.findFieldByName(self._flattenGroupName)
            if field.isValid():
                self.setFlattenGroup(field)

    def _defineDataProjectionOrientationField(self):
        coordinatesCount = self._modelCoordinatesField.getNumberOfComponents()
        self._dataProjectionOrientationField = findOrCreateFieldFiniteElement(
            self._fieldmodule, "data_projection_orientation",
            components_count=coordinatesCount*coordinatesCount)

    def defineDataProjectionFields(self):
        self._dataProjectionGroupNames = []
        self._dataProjectionNodeGroupFields = []
        self._dataProjectionNodesetGroups = []
        with ChangeManager(self._fieldmodule):
            datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
            for d in range(2):
                mesh = self.getMesh(d + 1)  # mesh1d, mesh2d
                group = self._fieldmodule.createFieldGroup()
                group.setName(getUniqueFieldName(self._fieldmodule, "data_projection_group_" + mesh.getName()))
                self._dataProjectionNodeGroupFields.append(group)
                self._dataProjectionNodesetGroups.append(group.createNodesetGroup(datapoints))
            self._defineDataProjectionOrientationField()

    def calculateGroupDataProjections(self, fieldcache, group, dataGroup, meshGroup, findHighestDimension, meshLocation,
                                      activeFitterStepConfig: FitterStepConfig):
        """
        Project data points for group. Assumes called while ChangeManager is active for fieldmodule.
        :param fieldcache: Fieldcache for zinc field evaluations in region.
        :param group: The FieldGroup being fitted (parent of dataGroup, meshGroup).
        :param dataGroup: Nodeset group containing data points to project.
        :param meshGroup: MeshGroup containing surfaces/lines to project onto.
        :param findHighestDimension: Set to True if mesh group does not have a parent/ancestor map to highest dimension
        model fit mesh, requiring an EXACT re-projection of the coordinates at the NEAREST location on meshGroup.
        :param meshLocation: FieldStoredMeshLocation to store found location in on highest dimension mesh.
        :param activeFitterStepConfig: Where to get current projection modes from.
        """
        groupName = group.getName()
        meshDimension = meshGroup.getDimension()
        dataProjectionNodesetGroup = self._dataProjectionNodesetGroups[meshDimension - 1]
        sizeBefore = dataProjectionNodesetGroup.getSize()
        dataCoordinates = self._dataCoordinatesField
        dataProportion = activeFitterStepConfig.getGroupDataProportion(groupName)[0]
        outlierLength = activeFitterStepConfig.getGroupOutlierLength(groupName)[0]
        maximumProjectionLength = 0.0
        dataProjectionLengths = []  # For relative outliers: list of (data identifier, projection length)
        centralProjection = activeFitterStepConfig.getGroupCentralProjection(groupName)[0]
        if centralProjection:
            # use centre of bounding box as middle of data; previous use of mean was affected by uneven density
            minDataCoordinates, maxDataCoordinates = evaluate_field_nodeset_range(dataCoordinates, dataGroup)
            if (minDataCoordinates is None) or (maxDataCoordinates is None):
                print("Error: Central projection failed to get mean coordinates of data for group " + groupName)
                return
            dataCentre = mult(add(minDataCoordinates, maxDataCoordinates), 0.5)
            # print("Centre Groups dataCentre", dataCentre)
            # get geometric centre of meshGroup
            meshGroupCoordinatesIntegral = self._fieldmodule.createFieldMeshIntegral(
                self._modelCoordinatesField, self._modelCoordinatesField, meshGroup)
            meshGroupCoordinatesIntegral.setNumbersOfPoints([3])
            meshGroupArea = self._fieldmodule.createFieldMeshIntegral(
                self._fieldmodule.createFieldConstant([1.0]), self._modelCoordinatesField, meshGroup)
            meshGroupArea.setNumbersOfPoints([3])
            result1, coordinatesIntegral = meshGroupCoordinatesIntegral.evaluateReal(
                fieldcache, self._modelCoordinatesField.getNumberOfComponents())
            result2, area = meshGroupArea.evaluateReal(fieldcache, 1)
            if (result1 != RESULT_OK) or (result2 != RESULT_OK) or (area <= 0.0):
                print("Error: Centre Groups projection failed to get mean coordinates of mesh for group " + groupName)
                return
            meshCentre = [s / area for s in coordinatesIntegral]
            # print("Centre Groups meshCentre", meshCentre)
            # offset dataCoordinates to make dataCentre coincide with meshCentre
            dataCoordinates = dataCoordinates + self._fieldmodule.createFieldConstant(sub(meshCentre, dataCentre))

        # find nearest locations on 1-D or 2-D feature but store on highest dimension mesh
        storeMesh = self.getHighestDimensionMesh()
        storeMeshDimension = storeMesh.getDimension()
        if self._modelFitGroup:
            storeMesh = self._modelFitGroup.getMeshGroup(storeMesh)
            assert storeMesh.isValid(), "Model fit group is wrong dimension"
        if findHighestDimension:
            # find nearest on search mesh, then find exact on store mesh from projected coordinates
            findLocation1 = self._fieldmodule.createFieldFindMeshLocation(
                dataCoordinates, self._modelCoordinatesField, meshGroup)
            findLocation1.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_NEAREST)
            projectedCoordinates = self._fieldmodule.createFieldEmbedded(self._modelCoordinatesField, findLocation1)
            findLocation = self._fieldmodule.createFieldFindMeshLocation(
                projectedCoordinates, self._modelCoordinatesField, storeMesh)
            findLocation.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_EXACT)
        else:
            # automatic map from search mesh to store mesh
            findLocation = self._fieldmodule.createFieldFindMeshLocation(
                dataCoordinates, self._modelCoordinatesField, storeMesh)
            assert RESULT_OK == findLocation.setSearchMesh(meshGroup)
            findLocation.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_NEAREST)
        nodeIter = dataGroup.createNodeiterator()
        node = nodeIter.next()
        dataProportionCounter = 0.5
        pointsProjected = 0
        outlierPointsRemoved = 0
        while node.isValid():
            dataProportionCounter += dataProportion
            if dataProportionCounter >= 1.0:
                dataProportionCounter -= 1.0
                fieldcache.setNode(node)
                element, xi = findLocation.evaluateMeshLocation(fieldcache, storeMeshDimension)
                if element.isValid():
                    result = meshLocation.assignMeshLocation(fieldcache, element, xi)
                    assert result == RESULT_OK, \
                        "Error: Failed to assign data projection mesh location for group " + groupName
                    result, projectionLength = self._dataErrorField.evaluateReal(fieldcache, 1)
                    if projectionLength > maximumProjectionLength:
                        maximumProjectionLength = projectionLength
                    if outlierLength < 0.0:
                        # store and filter once we know the maximum
                        dataProjectionLengths.append((node.getIdentifier(), projectionLength))
                    if (outlierLength <= 0.0) or (projectionLength <= outlierLength):
                        dataProjectionNodesetGroup.addNode(node)
                        pointsProjected += 1
                    else:
                        outlierPointsRemoved += 1
            node = nodeIter.next()
        if outlierLength < 0.0:
            relativeOutlierLength = (1.0 + outlierLength) * maximumProjectionLength
            for nodeIdentifier, projectionLength in dataProjectionLengths:
                if projectionLength > relativeOutlierLength:
                    node = dataGroup.findNodeByIdentifier(nodeIdentifier)
                    dataProjectionNodesetGroup.removeNode(node)
                    pointsProjected -= 1
                    outlierPointsRemoved += 1
        if self.getDiagnosticLevel() > 0:
            print(str(pointsProjected) + " of " + str(dataGroup.getSize()) + " data points projected for group " +
                  groupName + "; " + str(outlierPointsRemoved) + " outliers removed")
        # add to active group
        self._activeDataNodesetGroup.addNodesConditional(self._dataProjectionNodeGroupFields[meshDimension - 1])
        return

    def getGroupDataProjectionNodesetGroup(self, group: FieldGroup):
        """
        :return: Data NodesetGroup containing points for projection of group, otherwise None.
        """
        datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        dataNodesetGroup = group.getNodesetGroup(datapoints)
        if dataNodesetGroup.isValid() and (dataNodesetGroup.getSize() > 0):
            return dataNodesetGroup
        return None

    def getGroupDataProjectionMeshGroup(self, group: FieldGroup, fitterStep: FitterStep):
        """
        Get mesh group for 2D if not 1D mesh containing elements for projecting data in group, if any.
        If there is a subgroup set or inherited by the group, finds or builds on demand an equal or lower
        dimensional intersection mesh group.
        :group: Zinc annotation group for which to get projection mesh group.
        :fitterStep: FitterStep to get config for, with optional subgroup to intersect with.
        :return: MeshGroup or None if none or empty, findHighestDimension.
        The second argument is True if any of the mesh group doesn't have a child-parent[-grandparent]
        map to highest dimension elements in the model or model fit group, and hence projections must be
        re-found on the top-level mesh group.
        """
        groupName = group.getName()
        activeFitterStepConfig = self.getActiveFitterStepConfig(fitterStep)
        subgroup = activeFitterStepConfig.getGroupProjectionSubgroup(groupName)[0]
        groupProjectionData = self._groupProjectionData.get(groupName)
        if groupProjectionData:
            if (groupProjectionData[0] == subgroup):
                return groupProjectionData[1], groupProjectionData[2]
        returnMeshGroup = None
        findHighestDimension = False
        with ChangeManager(self._fieldmodule):
            if groupProjectionData:
                del self._groupProjectionData[groupName]  # cleans up field references if subgroup changed
            highestDimensionMesh = self.getHighestDimensionMesh()
            highestDimension = highestDimensionMesh.getDimension()

            for meshDimension in range(2, 0, -1):
                if meshDimension > highestDimension:
                    continue
                mesh = self.getMesh(meshDimension)
                meshGroup = group.getMeshGroup(mesh)
                if meshGroup.isValid() and (meshGroup.getSize() > 0):
                    if subgroup:
                        subgroupMeshGroup = subgroup.getMeshGroup(mesh)
                        if subgroupMeshGroup.isValid() and (subgroupMeshGroup.getSize() > 0):
                            intersectionGroup = self._fieldmodule.createFieldGroup()
                            intersectionGroup.setName(groupName + " " + subgroup.getName())  # for debugging
                            returnMeshGroup = intersectionGroup.createMeshGroup(mesh)
                            returnMeshGroup.addElementsConditional(self._fieldmodule.createFieldAnd(group, subgroup))
                            if returnMeshGroup.getSize() > 0:
                                break
                            del intersectionGroup
                            returnMeshGroup = None
                    else:
                        returnMeshGroup = meshGroup
                        break
        if returnMeshGroup:
            # do any elements in returnMeshGroup NOT have self or an ancestor in modelFitMesh?
            # If so, need to find highest dimension mesh location by a second FindMeshLocation field when projecting
            modelFitMesh = highestDimensionMesh
            if self._modelFitGroup:
                modelFitMesh = self._modelFitGroup.getMeshGroup(highestDimensionMesh)
            elementiterator = returnMeshGroup.createElementiterator()
            element = elementiterator.next()
            while element.isValid():
                if not element_or_ancestor_is_in_mesh(element, modelFitMesh):
                    findHighestDimension = True
                    break
                element = elementiterator.next()
        self._groupProjectionData[groupName] = (subgroup, returnMeshGroup, findHighestDimension)

        return returnMeshGroup, findHighestDimension

    def calculateDataProjections(self, fitterStep: FitterStep):
        """
        Find projections of datapoints' coordinates onto model coordinates,
        by groups i.e. from datapoints group onto matching 2-D or 1-D mesh group.
        Calculate and store projection direction unit vector.
        """
        assert self._dataCoordinatesField and self._modelCoordinatesField
        activeFitterStepConfig = self.getActiveFitterStepConfig(fitterStep)
        with ChangeManager(self._fieldmodule):
            # build group of active data and marker points
            self._activeDataNodesetGroup.removeAllNodes()
            if self._markerDataLocationGroupField:
                self._activeDataNodesetGroup.addNodesConditional(self._markerDataLocationGroupField)
            # build groups of data and elements participating in projections in 1 and 2 dimension
            for d in range(2):
                self._dataProjectionNodesetGroups[d].removeAllNodes()
                self._activeDataProjectionMeshGroups[d].removeAllElements()

            datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
            fieldcache = self._fieldmodule.createFieldcache()
            groups = getGroupList(self._fieldmodule)
            for group in groups:
                if not group.isManaged():
                    continue  # skip cmiss_selection, for example
                groupName = group.getName()
                dataGroup = self.getGroupDataProjectionNodesetGroup(group)
                if not dataGroup:
                    continue
                meshGroup, findHighestDimension = self.getGroupDataProjectionMeshGroup(group, fitterStep)
                if not meshGroup:
                    if self.getDiagnosticLevel() > 0:
                        if group != self._markerGroup:
                            print("Warning: Cannot project data for group " + groupName + " as no matching mesh group")
                    continue
                if groupName not in self._dataProjectionGroupNames:
                    self._dataProjectionGroupNames.append(groupName)  # so only define mesh location, or warn once
                    fieldcache.setNode(dataGroup.createNodeiterator().next())
                    if not self._dataCoordinatesField.isDefinedAtLocation(fieldcache):
                        if self.getDiagnosticLevel() > 0:
                            print("Warning: Cannot project data for group " + groupName +
                                  " as field " + self._dataCoordinatesField.getName() + " is not defined on data")
                        continue
                    # define host location, data point weight and data projection orientation on data Group:
                    nodetemplate = datapoints.createNodetemplate()
                    nodetemplate.defineField(self._dataHostLocationField)
                    # need to define storage for marker data weight, but don't assign here
                    nodetemplate.defineField(self._dataWeightField)
                    nodetemplate.defineField(self._dataProjectionOrientationField)
                    nodeIter = dataGroup.createNodeiterator()
                    node = nodeIter.next()
                    while node.isValid():
                        node.merge(nodetemplate)
                        node = nodeIter.next()
                    del nodetemplate
                self.calculateGroupDataProjections(fieldcache, group, dataGroup, meshGroup, findHighestDimension,
                                                   self._dataHostLocationField, activeFitterStepConfig)
                # add elements being projected onto to active group for mesh dimension
                self._activeDataProjectionMeshGroups[meshGroup.getDimension() - 1].addElementsConditional(group)

            # Assign data projection orientation
            coordinatesCount = self._modelCoordinatesField.getNumberOfComponents()
            highestMeshDimension = self.getHighestDimensionMesh().getDimension()
            for meshDimension in range(1, 3):
                nodesetGroup = self._dataProjectionNodesetGroups[meshDimension - 1]
                if nodesetGroup.getSize() > 0:
                    if meshDimension == highestMeshDimension:
                        faceLocationField = self._dataHostLocationField  # 2-D fit case
                    else:
                        # unfortunately need to find mesh location again, this time on active line/face mesh
                        faceLocationField = self._fieldmodule.createFieldFindMeshLocation(
                            self._dataHostCoordinatesField, self._modelCoordinatesField,
                            self._activeDataProjectionMeshGroups[meshDimension - 1])
                        # in theory SEARCH_MODE_EXACT should work, however tests fail
                        faceLocationField.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_NEAREST)
                    d1 = self._fieldmodule.createFieldDerivative(self._modelCoordinatesField, 1)
                    if meshDimension == 1:
                        d1 = self._fieldmodule.createFieldNormalise(d1)
                        if coordinatesCount == 3:
                            side1 = self._fieldmodule.createFieldCrossProduct(
                                self._fieldmodule.createFieldConstant([1.0, 0.0, 0.0]), d1)
                            side2 = self._fieldmodule.createFieldCrossProduct(
                                self._fieldmodule.createFieldConstant([0.0, 1.0, 0.0]), d1)
                            side1 = self._fieldmodule.createFieldNormalise(
                                self._fieldmodule.createFieldIf(
                                    self._fieldmodule.createFieldGreaterThan(
                                        self._fieldmodule.createFieldMagnitude(side1),
                                        self._fieldmodule.createFieldMagnitude(side2)),
                                    side1, side2))
                            side2 = self._fieldmodule.createFieldCrossProduct(d1, side1)
                            directions = [d1, side1, side2]
                            del side2
                            del side1
                        elif coordinatesCount == 2:
                            side1 = self._fieldmodule.createFieldMatrixMultiply(
                                2, self._fieldmodule.createFieldConstant([0.0, -1.0, 1.0, 0.0]), d1)
                            directions = [d1, side1]
                            del side1
                        else:  # coordinatesCount == 1:
                            directions = [d1]
                        sourceOrientationField = self._fieldmodule.createFieldEmbedded(
                            self._fieldmodule.createFieldConcatenate(directions), faceLocationField)
                        del directions
                    else:  # meshDimension == 2:
                        # if data delta has a non-negligible tangential component,
                        # align d1 with it to enable stretch in that direction
                        d2 = self._fieldmodule.createFieldDerivative(self._modelCoordinatesField, 2)
                        dLimit = self._fieldmodule.createFieldConstant([1.0E-5]) * \
                                     self._fieldmodule.createFieldMagnitude(
                                        self._fieldmodule.createFieldEmbedded(d1 + d2, faceLocationField))
                        if coordinatesCount == 3:
                            normal = self._fieldmodule.createFieldEmbedded(
                                self._fieldmodule.createFieldNormalise(
                                    self._fieldmodule.createFieldCrossProduct(d1, d2)),
                                faceLocationField)
                            magDeltaNormal = self._fieldmodule.createFieldDotProduct(
                                self._dataDeltaField, normal)
                            deltaNormal = magDeltaNormal * normal
                            deltaTangent = self._dataDeltaField - deltaNormal
                            d1 = self._fieldmodule.createFieldNormalise(
                                self._fieldmodule.createFieldIf(
                                    self._fieldmodule.createFieldGreaterThan(
                                        self._fieldmodule.createFieldMagnitude(deltaTangent), dLimit),
                                    deltaTangent,
                                    self._fieldmodule.createFieldEmbedded(d1, faceLocationField)))
                            d2 = self._fieldmodule.createFieldCrossProduct(normal, d1)
                            directions = [d1, d2, normal]
                            del deltaTangent
                            del deltaNormal
                            del magDeltaNormal
                            del normal
                        else:  # coordinatesCount == 2:
                            d1 = self._fieldmodule.createFieldNormalise(
                                self._fieldmodule.createFieldIf(
                                    self._fieldmodule.createFieldGreaterThan(
                                        self._fieldmodule.createFieldMagnitude(self._dataDeltaField), dLimit),
                                    self._dataDeltaField,
                                    self._fieldmodule.createFieldEmbedded(d1, faceLocationField)))
                            d2 = self._fieldmodule.createFieldMatrixMultiply(
                                2, self._fieldmodule.createFieldConstant([0.0, -1.0, 1.0, 0.0]), d1)
                            directions = [d1, d2]
                        del dLimit
                        del d2
                        sourceOrientationField = self._fieldmodule.createFieldConcatenate(directions)
                        del directions
                    del d1
                    del faceLocationField
                    fieldassignment = self._dataProjectionOrientationField.createFieldassignment(sourceOrientationField)
                    fieldassignment.setNodeset(nodesetGroup)
                    result = fieldassignment.assign()
                    assert result in [RESULT_OK, RESULT_WARNING_PART_DONE], \
                        "Error:  Failed to assign data projection orientation for mesh dimension " + str(meshDimension)
                    del fieldassignment
                    del sourceOrientationField

            if self.getDiagnosticLevel() > 0:
                # Warn about unprojected points
                unprojectedDataGroup = self._fieldmodule.createFieldGroup()
                unprojectedDataNodesetGroup = unprojectedDataGroup.createNodesetGroup(datapoints)
                unprojectedDataNodesetGroup.addNodesConditional(
                    self._fieldmodule.createFieldIsDefined(self._dataCoordinatesField))
                for d in range(2):
                    unprojectedDataNodesetGroup.removeNodesConditional(self._dataProjectionNodeGroupFields[d])
                unprojectedCount = unprojectedDataNodesetGroup.getSize()
                if unprojectedCount > 0:
                    print("Warning: " + str(unprojectedCount) +
                          " data points with data coordinates have not been projected")
                del unprojectedDataNodesetGroup
                del unprojectedDataGroup

                # Write projection error measures
                rmsErrorValue, maxErrorValue = self.getDataRMSAndMaximumProjectionError()
                print("Data projection RMS error", rmsErrorValue, "Max error", maxErrorValue)

            # remove temporary objects before ChangeManager exits
            del fieldcache

    def getDataProjectionOrientationField(self):
        return self._dataProjectionOrientationField

    def getDataProjectionGroupNames(self):
        return self._dataProjectionGroupNames

    def getDataProjectionNodeGroupField(self, meshDimension):
        assert 1 <= meshDimension <= 2
        return self._dataProjectionNodeGroupFields[meshDimension - 1]

    def getDataProjectionNodesetGroup(self, meshDimension):
        assert 1 <= meshDimension <= 2
        return self._dataProjectionNodesetGroups[meshDimension - 1]

    def getMarkerDataLocationGroupField(self):
        return self._markerDataLocationGroupField

    def getMarkerDataLocationNodesetGroup(self):
        return self._markerDataLocationGroup

    def getMarkerDataLocationField(self):
        """
        Same as for all other data points.
        """
        return self._dataHostLocationField

    def getContext(self):
        return self._context

    def getZincVersion(self):
        """
        :return: zinc version numbers [major, minor, patch].
        """
        return self._zincVersion

    def getRegion(self):
        return self._region

    def getFieldmodule(self):
        return self._fieldmodule

    def getFitterSteps(self):
        return self._fitterSteps

    def getMesh(self, dimension):
        """
        :param dimension: Mesh dimension.
        :return: Zinc Mesh; invalid if dimension not from 1 to 3.
        """
        return self._fieldmodule.findMeshByDimension(dimension)

    def getHighestDimensionMesh(self):
        """
        :return: Highest dimension mesh with elements in it, or None if none.
        """
        for dimension in range(3, 0, -1):
            mesh = self._fieldmodule.findMeshByDimension(dimension)
            if mesh.getSize() > 0:
                return mesh
        return None

    def _log_message_type_to_text(self, message_type):
        # 'MESSAGE_TYPE_ERROR', 'MESSAGE_TYPE_INFORMATION', 'MESSAGE_TYPE_INVALID', 'MESSAGE_TYPE_WARNING'
        if self._logger.MESSAGE_TYPE_ERROR == message_type:
            return "Error"
        if self._logger.MESSAGE_TYPE_INFORMATION == message_type:
            return "Information"
        if self._logger.MESSAGE_TYPE_WARNING == message_type:
            return "Warning"

        return "Invalid"

    def print_log(self):
        loggerMessageCount = self._logger.getNumberOfMessages()
        if loggerMessageCount > 0:
            for i in range(1, loggerMessageCount + 1):
                print(f"[Message {i}] {self._log_message_type_to_text(self._logger.getMessageTypeAtIndex(i))}: {self._logger.getMessageTextAtIndex(i)}")
            self._logger.removeAllMessages()

    def getDiagnosticLevel(self):
        return self._diagnosticLevel

    def setDiagnosticLevel(self, diagnosticLevel):
        """
        :param diagnosticLevel: 0 = no diagnostic messages. 1 = Information and warning messages.
        2 = Also optimisation reports.
        """
        assert diagnosticLevel >= 0
        self._diagnosticLevel = diagnosticLevel

    def updateModelReferenceCoordinates(self):
        assignFieldParameters(self._modelReferenceCoordinatesField, self._modelCoordinatesField)

    def writeModel(self, modelFileName):
        """
        Write model nodes and elements with model coordinates field to file.
        Note: Output field name is prefixed with "fitted ".
        """
        with ChangeManager(self._fieldmodule):
            # temporarily rename model coordinates field to prefix with "fitted "
            # so can be used along with original coordinates in later steps
            outputCoordinatesFieldName = "fitted " + self._modelCoordinatesFieldName
            self._modelCoordinatesField.setName(outputCoordinatesFieldName)

            sir = self._region.createStreaminformationRegion()
            sir.setRecursionMode(sir.RECURSION_MODE_OFF)
            srf = sir.createStreamresourceFile(modelFileName)
            sir.setResourceFieldNames(srf, [outputCoordinatesFieldName])
            sir.setResourceDomainTypes(srf, Field.DOMAIN_TYPE_NODES |
                                       Field.DOMAIN_TYPE_MESH1D | Field.DOMAIN_TYPE_MESH2D | Field.DOMAIN_TYPE_MESH3D)
            if self._modelFitGroup:
                sir.setResourceGroupName(srf, self._modelFitGroup.getName())
            result = self._region.write(sir)
            # self.print_log()

            # restore original name
            self._modelCoordinatesField.setName(self._modelCoordinatesFieldName)

            assert result == RESULT_OK

    def writeData(self, fileName):
        sir = self._region.createStreaminformationRegion()
        sir.setRecursionMode(sir.RECURSION_MODE_OFF)
        sr = sir.createStreamresourceFile(fileName)
        sir.setResourceDomainTypes(sr, Field.DOMAIN_TYPE_DATAPOINTS)
        self._region.write(sir)
