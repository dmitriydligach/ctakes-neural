package org.apache.ctakes.thyme.keras;

import java.io.File;
import java.io.FileNotFoundException;

import org.apache.ctakes.temporal.keras.KerasStringOutcomeClassifierBuilder;
import org.apache.ctakes.temporal.keras.ScriptStringOutcomeDataWriter;

import com.google.common.annotations.Beta;

/**
 * <br>
 * Copyright (c) 2016, Regents of the University of Colorado <br>
 * All rights reserved.
 * 
 * @author Tim Miller
 * @version 2.0.1
 * 
 */
@Beta
public class KerasStringFeatureDataWriter extends ScriptStringOutcomeDataWriter<KerasStringOutcomeClassifierBuilder>{

  public KerasStringFeatureDataWriter(File outputDirectory)
      throws FileNotFoundException {
    super(outputDirectory);
  }

  @Override
  protected KerasStringOutcomeClassifierBuilder newClassifierBuilder() {
    return new KerasStringOutcomeClassifierBuilder();
  }
}
