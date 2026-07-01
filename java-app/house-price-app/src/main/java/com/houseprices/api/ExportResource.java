package com.houseprices.api;

import com.houseprices.ingest.PropertyRecord;
import com.houseprices.service.PropertyStore;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;
import org.jboss.logging.Logger;

import java.io.*;
import java.util.List;

/**
 * GET /api/export/csv?source=all|sales|rentcast|zillow
 * Streams all in-memory records as a CSV download.
 */
@Path("/api/export")
public class ExportResource {

    private static final Logger LOG = Logger.getLogger(ExportResource.class);

    @Inject PropertyStore store;

    @GET
    @Path("/csv")
    @Produces("text/csv")
    public Response exportCsv(
            @QueryParam("source") @DefaultValue("all") String source) {

        List<PropertyRecord> records = switch (source) {
            case "sales"    -> store.getBySource("sales");
            case "rentcast" -> store.getBySource("rentcast");
            case "zillow"   -> store.getBySource("zillow");
            default         -> store.getAll();
        };

        LOG.infof("Exporting %d records (source=%s)", records.size(), source);

        StreamingOutput stream = out -> {
            try (PrintWriter pw = new PrintWriter(new BufferedWriter(new OutputStreamWriter(out)))) {
                pw.println("id,address,source,lat,lng,h3Index,community," +
                           "sqft,sqftLot,beds,baths,homeType," +
                           "saleDate,salePrice,predictedPrice,pctError,listingUrl");
                for (PropertyRecord r : records) {
                    pw.printf("%s,%s,%s,%.6f,%.6f,%s,%s,%.0f,%.0f,%d,%.1f,%s,%s,%.0f,%.0f,%.2f,%s%n",
                        esc(r.id()), esc(r.address()), esc(r.source()),
                        r.lat(), r.lng(),
                        esc(r.h3Index()), esc(r.community()),
                        r.sqft(), r.sqftLot(), r.beds(), r.baths(),
                        esc(r.homeType()),
                        r.saleDate() != null ? r.saleDate().toString() : "",
                        r.salePrice(), r.predictedPrice(), r.pctError(),
                        esc(r.listingUrl())
                    );
                }
            }
        };

        String filename = "house_prices_" + source + ".csv";
        return Response.ok(stream)
            .header("Content-Disposition", "attachment; filename=\"" + filename + "\"")
            .build();
    }

    private String esc(String s) {
        if (s == null) return "";
        return (s.contains(",") || s.contains("\"") || s.contains("\n"))
            ? "\"" + s.replace("\"", "\"\"") + "\"" : s;
    }
}
