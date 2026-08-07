package com.houseprices.api;

import com.houseprices.service.PropertyStore;
import io.quarkus.runtime.StartupEvent;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

/**
 * Starts with an empty property store. Source data is loaded on demand by
 * {@link com.houseprices.service.LiveSourceLoader} when a dependent map layer
 * is selected.
 */
@ApplicationScoped
public class StartupLoader {

    private static final Logger LOG = Logger.getLogger(StartupLoader.class);

    @Inject PropertyStore        store;

    void onStart(@Observes StartupEvent ev) {
        LOG.infof("Startup complete. Source records load on demand. Total records: %d | %s",
            store.totalCount(), store.countsBySource());
    }
}
